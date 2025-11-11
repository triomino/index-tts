import os
import time
import torch
import math
from torch import Tensor
import torchaudio
import warnings
from typing import Callable, Optional, Tuple, Union
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import random
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from infer_v2 import IndexTTS2, find_most_similar_cosine
from indextts.s2mel.modules.diffusion_transformer import DiT, sequence_mask
from indextts.gpt.model_v2 import GPT2InferenceModel, LearnedPositionEmbeddings
from indextts.s2mel.modules.gpt_fast.model import Attention, apply_rotary_emb
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.models.gpt2.modeling_gpt2 import eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

import torch_npu
from torch_npu.contrib import transfer_to_npu
import torchair

os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache' # TODO: remove

# 批次大小列表，开启 auto_split 时只会用这些大小的 batch size，减少因为 batch size 变化导致重编译次数。
DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 48, 64]

# 音频切割后对齐到 batch 内最大长度
def make_batch(self, sentences, pad_token_id):
    text_tokens = [self.tokenizer.convert_tokens_to_ids(sent) for sent in sentences]
    lens = torch.tensor([len(tokens) for tokens in text_tokens],dtype=torch.int32, device=self.device)
    max_len = max(lens)
    padded_tokens = [tokens + [pad_token_id] * (max_len - len(tokens)) for tokens in text_tokens]
    return torch.tensor(padded_tokens, dtype=torch.int32, device=self.device), lens

def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), 1, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), 1).to(dtype)

def _prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length):
    combined_attention_mask=None
    if input_shape[-1] > 1:
        combined_attention_mask=_make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length
        )
    if attention_mask is not None:
        expanded_attn_mask=_expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(inputs_embeds.device)
        combined_attention_mask=(
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )
    return combined_attention_mask

class IndexTTS2NPU(IndexTTS2):
    def __init__(self, static=False, *args, **kwargs):
        """
        继承自 IndexTTS2，并添加一个新的 'static' 启动参数。
        
        Args:
            static (bool): 是否开启静态图模式，默认为 False。静态图在 batch size 改变时会重编译，因此建议推理时开启 auto_split 自动对齐到预编译好的 batch size.
        """
        self.static = static
        super().__init__(*args, **kwargs)
        # GPT 模块传递静态图模式参数
        self.gpt.static = static
        for block in self.gpt.inference_model.transformer.h:
            block.attn.static = static
        # DiT 模块传递静态图模式参数
        self.s2mel.models['cfm'].estimator.static=static
        self.s2mel.models['cfm'].estimator.max_seq_len=2048
        self.s2mel.models['cfm'].estimator.padding_token=8193 # TODO: 改成 stop_mel_token
        # 改为字典结构，用于存储不同 batch_size 的编译图
        self.s2mel.models['cfm'].estimator.compiled_transformers = {}

        self.gpt.inference_model.static = self.static
        # 改为字典结构，用于存储不同 batch_size 的编译图
        self.gpt.inference_model.compiled_transformers = {}

def infer_fast(self, spk_audio_prompt, text, output_path,
            emo_audio_prompt=None, emo_alpha=1.0,
            emo_vector=None,
            use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
            verbose=False, max_text_tokens_per_segment=120, 
            auto_split=False,
            **generation_kwargs):
    """批量推理。基于 infer_v2.py，将音频切割后推理，在全流程做了 batch 化。
    Args:
        auto_split (bool): 是否自动切割输入文本。开启后 max_text_tokens_per_segment 无效，自动切割到合适的 batch_size 上。
    """
    # max_text_tokens_per_segment 的上下限
    MIN_TOKENS_PER_SEGMENT = 30
    MAX_TOKENS_PER_SEGMENT = 200
    
    print(">> starting inference...")
    self._set_gr_progress(0, "starting inference...")
    if verbose:
        print(f"origin text:{text}, spk_audio_prompt:{spk_audio_prompt}, "
                f"emo_audio_prompt:{emo_audio_prompt}, emo_alpha:{emo_alpha}, "
                f"emo_vector:{emo_vector}, use_emo_text:{use_emo_text}, "
                f"emo_text:{emo_text}")
    start_time = time.perf_counter()

    if use_emo_text or emo_vector is not None:
        # we're using a text or emotion vector guidance; so we must remove
        # "emotion reference voice", to ensure we use correct emotion mixing!
        emo_audio_prompt = None

    if use_emo_text:
        # automatically generate emotion vectors from text prompt
        if emo_text is None:
            emo_text = text  # use main text prompt
        emo_dict = self.qwen_emo.inference(emo_text)
        print(f"detected emotion vectors from text: {emo_dict}")
        # convert ordered dict to list of vectors; the order is VERY important!
        emo_vector = list(emo_dict.values())

    if emo_vector is not None:
        # we have emotion vectors; they can't be blended via alpha mixing
        # in the main inference process later, so we must pre-calculate
        # their new strengths here based on the alpha instead!
        emo_vector_scale = max(0.0, min(1.0, emo_alpha))
        if emo_vector_scale != 1.0:
            # scale each vector and truncate to 4 decimals (for nicer printing)
            emo_vector = [int(x * emo_vector_scale * 10000) / 10000 for x in emo_vector]
            print(f"scaled emotion vectors to {emo_vector_scale}x: {emo_vector}")

    if emo_audio_prompt is None:
        # we are not using any external "emotion reference voice"; use
        # speaker's voice as the main emotion reference audio.
        emo_audio_prompt = spk_audio_prompt
        # must always use alpha=1.0 when we don't have an external reference voice
        emo_alpha = 1.0

    # 如果参考音频改变了，才需要重新生成, 提升速度
    if self.cache_spk_cond is None or self.cache_spk_audio_prompt != spk_audio_prompt:
        if self.cache_spk_cond is not None:
            self.cache_spk_cond = None
            self.cache_s2mel_style = None
            self.cache_s2mel_prompt = None
            self.cache_mel = None
            torch.cuda.empty_cache()
        audio, sr = self._load_and_cut_audio(spk_audio_prompt, 15, verbose)
        audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

        inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"]
        attention_mask = inputs["attention_mask"]
        input_features = input_features.to(self.device)
        attention_mask = attention_mask.to(self.device)
        spk_cond_emb = self.get_emb(input_features, attention_mask)

        _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
        ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
        ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
        feat = torchaudio.compliance.kaldi.fbank(audio_16k.to(ref_mel.device),
                                                    num_mel_bins=80,
                                                    dither=0,
                                                    sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)  # feat2另外一个滤波器能量组特征[922, 80]
        style = self.campplus_model(feat.unsqueeze(0))  # 参考音频的全局style2[1,192]

        prompt_condition = self.s2mel.models['length_regulator'](S_ref,
                                                                    ylens=ref_target_lengths,
                                                                    n_quantizers=3,
                                                                    f0=None)[0]

        self.cache_spk_cond = spk_cond_emb
        self.cache_s2mel_style = style
        self.cache_s2mel_prompt = prompt_condition
        self.cache_spk_audio_prompt = spk_audio_prompt
        self.cache_mel = ref_mel
    else:
        style = self.cache_s2mel_style
        prompt_condition = self.cache_s2mel_prompt
        spk_cond_emb = self.cache_spk_cond
        ref_mel = self.cache_mel

    if emo_vector is not None:
        weight_vector = torch.tensor(emo_vector).to(self.device)
        if use_random:
            random_index = [random.randint(0, x - 1) for x in self.emo_num]
        else:
            random_index = [find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]

        emo_matrix = [tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix)]
        emo_matrix = torch.cat(emo_matrix, 0)
        emovec_mat = weight_vector.unsqueeze(1) * emo_matrix
        emovec_mat = torch.sum(emovec_mat, 0)
        emovec_mat = emovec_mat.unsqueeze(0)

    if self.cache_emo_cond is None or self.cache_emo_audio_prompt != emo_audio_prompt:
        if self.cache_emo_cond is not None:
            self.cache_emo_cond = None
            torch.cuda.empty_cache()
        emo_audio, _ = self._load_and_cut_audio(emo_audio_prompt, 15, verbose, sr=16000)
        emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
        emo_input_features = emo_inputs["input_features"]
        emo_attention_mask = emo_inputs["attention_mask"]
        emo_input_features = emo_input_features.to(self.device)
        emo_attention_mask = emo_attention_mask.to(self.device)
        emo_cond_emb = self.get_emb(emo_input_features, emo_attention_mask)

        self.cache_emo_cond = emo_cond_emb
        self.cache_emo_audio_prompt = emo_audio_prompt
    else:
        emo_cond_emb = self.cache_emo_cond

    self._set_gr_progress(0.1, "text processing...")
    text_tokens_list = self.tokenizer.tokenize(text)
    
    # 根据auto_split参数决定是否自动切割
    if auto_split:
        # 计算文本总token数
        total_tokens = len(text_tokens_list)
        
        # 初始化max_text_tokens_per_segment为上下限中的最大值
        current_max_tokens = min(MAX_TOKENS_PER_SEGMENT, total_tokens)
        
        # 二分查找最佳的max_text_tokens_per_segment
        best_segments = None
        best_batch_size = None
        
        low, high = MIN_TOKENS_PER_SEGMENT, min(MAX_TOKENS_PER_SEGMENT, total_tokens)
        while low <= high:
            mid = (low + high) // 2
            segments = self.tokenizer.split_segments(text_tokens_list, mid)
            num_segments = len(segments)
            
            # 如果段数在DEFAULT_BATCH_SIZES中，或者超过最大批次大小，则记录
            if num_segments in DEFAULT_BATCH_SIZES or num_segments > max(DEFAULT_BATCH_SIZES):
                best_segments = segments
                best_batch_size = num_segments
                high = mid - 1  # 尝试使用更小的segment
            else:
                low = mid + 1  # 尝试使用更大的segment
                
        # 如果没有找到合适的，使用最大值
        if best_segments is None:
            best_segments = self.tokenizer.split_segments(text_tokens_list, MIN_TOKENS_PER_SEGMENT)
            best_batch_size = len(best_segments)
            
        segments = best_segments
    else:
        # 不自动切割，使用用户指定的max_text_tokens_per_segment
        segments = self.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment)
        best_batch_size = len(segments)
    
    # 处理批次对齐
    if auto_split:
        # 确定目标批次大小
        target_batch_size = None
        
        # 如果段数在DEFAULT_BATCH_SIZES中，直接使用
        if best_batch_size in DEFAULT_BATCH_SIZES:
            target_batch_size = best_batch_size
        else:
            # 找到最接近且不小于当前段数的批次大小
            for bs in DEFAULT_BATCH_SIZES:
                if bs >= best_batch_size:
                    target_batch_size = bs
                    break
            
            # 如果没找到，使用最大的批次大小
            if target_batch_size is None:
                target_batch_size = max(DEFAULT_BATCH_SIZES)
        
        # 如果段数超过最大批次大小，进行多轮处理
        if best_batch_size > max(DEFAULT_BATCH_SIZES):
            # 分成多个批次，每批最多max(DEFAULT_BATCH_SIZES)个段
            batches = []
            for i in range(0, len(segments), max(DEFAULT_BATCH_SIZES)):
                batch_segments = segments[i:i+max(DEFAULT_BATCH_SIZES)]
                batch_size = len(batch_segments)
                
                # 填充到目标批次大小（使用最后一个段）
                if batch_size not in DEFAULT_BATCH_SIZES:
                    # 找到最接近且不小于当前批次大小的目标大小
                    for bs in DEFAULT_BATCH_SIZES:
                        if bs >= batch_size:
                            target_size = bs
                            break
                    
                    # 填充
                    padding_size = target_size - batch_size
                    if padding_size > 0:
                        # 使用最后一个段进行填充
                        last_segment = batch_segments[-1]
                        # 创建填充段（保证生成序列不长于其他序列）
                        padding_segments = [last_segment[:len(last_segment)-i] for i in range(padding_size)]
                        batch_segments.extend(padding_segments)
                
                batches.append((batch_segments, batch_size))
        else:
            # 单个批次，需要填充到目标批次大小
            padding_size = target_batch_size - best_batch_size
            if padding_size > 0:
                # 使用最后一个段进行填充
                last_segment = segments[-1]
                # 创建填充段（保证生成序列不长于其他序列）
                padding_segments = [last_segment[:len(last_segment)-i] for i in range(padding_size)]
                segments.extend(padding_segments)
            batches = [(segments, best_batch_size)]
    else:
        # 不自动切割，不进行填充
        batches = [(segments, best_batch_size)]
    
    segments_count = sum([len(batch[0]) for batch in batches])
    if verbose:
        print("text_tokens_list:", text_tokens_list)
        print("segments count:", segments_count)
        print("max_text_tokens_per_segment:", max_text_tokens_per_segment)
        print(*segments, sep="\n")
    if auto_split:
        print(f"auto_split enabled. Using {len(batches)} batches with sizes: {[len(batch[0]) for batch in batches]}")
    do_sample = generation_kwargs.pop("do_sample", True)
    top_p = generation_kwargs.pop("top_p", 0.8)
    top_k = generation_kwargs.pop("top_k", 30)
    temperature = generation_kwargs.pop("temperature", 0.8)
    autoregressive_batch_size = 1
    length_penalty = generation_kwargs.pop("length_penalty", 0.0)
    num_beams = generation_kwargs.pop("num_beams", 3)
    repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
    max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 1500)
    sampling_rate = 22050

    wavs = []
    gpt_gen_time = 0
    gpt_forward_time = 0
    s2mel_time = 0
    bigvgan_time = 0
    has_warned = False

    # 处理每个批次
    for batch_idx, (batch_segments, original_size) in enumerate(batches):
        self._set_gr_progress(0.2 + 0.7 * batch_idx / len(batches),
                            f"processing batch {batch_idx + 1}/{len(batches)}...")
        
        text_tokens, text_token_lens = self.make_batch(batch_segments, self.cfg.gpt.stop_text_token)
        
        if verbose:
            print(f"Batch {batch_idx + 1}: text_tokens shape: {text_tokens.shape}")
        
        m_start_time = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                emovec = self.gpt.merge_emovec(
                    spk_cond_emb,
                    emo_cond_emb,
                    torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                    torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                    alpha=emo_alpha
                )

                if emo_vector is not None:
                    emovec = emovec_mat + (1 - torch.sum(weight_vector)) * emovec

                b_text = text_tokens.size(0)
                if spk_cond_emb.shape[0] == 1 and b_text > 1:
                    spk_cond_emb = spk_cond_emb.repeat(b_text, 1, 1) # B,D,freams
                    cond_lengths = torch.full((b_text,),spk_cond_emb.shape[-1], device=text_tokens.device)
                else:
                    cond_lengths = torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device)
                if emo_cond_emb.shape[0] == 1 and b_text > 1:
                    emo_cond_emb = emo_cond_emb.repeat(b_text, 1, 1)
                    emo_cond_lengths=torch.full((b_text,),emo_cond_emb.shape[-1], device=text_tokens.device)
                else:
                    emo_cond_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device)
                if emovec.shape[0] == 1 and b_text > 1:
                    emovec = emovec.repeat(b_text, 1)
                
                codes, speech_conditioning_latent = self.gpt.inference_speech(
                    spk_cond_emb,
                    text_tokens,
                    emo_cond_emb,
                    cond_lengths=cond_lengths,
                    emo_cond_lengths=emo_cond_lengths,
                    emo_vec=emovec,
                    do_sample=True,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=autoregressive_batch_size,
                    length_penalty=length_penalty,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    max_generate_length=max_mel_tokens,
                    **generation_kwargs
                )

            gpt_gen_time += time.perf_counter() - m_start_time
            if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                warnings.warn(
                    f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                    f"Input text tokens: {text_tokens.shape[1]}. ",
                    category=RuntimeWarning
                )
                has_warned = True

            code_lens = []
            for code in codes:
                if self.stop_mel_token not in code:
                    code_len = len(code)
                else:
                    len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0] + 1
                    code_len = len_ - 1
                code_lens.append(code_len)
            code_lens = torch.LongTensor(code_lens).to(self.device)

            m_start_time = time.perf_counter()
            use_speed = torch.zeros(spk_cond_emb.size(0)).to(spk_cond_emb.device).long()
            with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                latent = self.gpt(
                    speech_conditioning_latent,
                    text_tokens,
                    text_token_lens,
                    codes,
                    code_lens,
                    emo_cond_emb,
                    cond_mel_lengths=torch.full((spk_cond_emb.shape[0],), spk_cond_emb.shape[-1], device=text_tokens.device),
                    emo_cond_mel_lengths=torch.full((emo_cond_emb.shape[0],),emo_cond_emb.shape[-1], device=text_tokens.device),
                    emo_vec=emovec,
                    use_speed=use_speed,
                )
                gpt_forward_time += time.perf_counter() - m_start_time
                
            dtype = None
            with torch.amp.autocast(text_tokens.device.type, enabled=dtype is not None, dtype=dtype):
                m_start_time = time.perf_counter()
                diffusion_steps = 25
                inference_cfg_rate = 0.7
                latent = self.s2mel.models['gpt_layer'](latent) # B,T,D
                S_infer = torch.cat([self.semantic_codec.quantizer.vq2emb(codes[i:i+1,:].unsqueeze(1)) for i in range(codes.size(0))],0)
                S_infer = S_infer.transpose(1, 2) # B,T,D
                S_infer = S_infer + latent
                target_lengths = (code_lens * 1.72).long()
                prompt_condition = prompt_condition.repeat(S_infer.size(0),1,1)
                cond = self.s2mel.models['length_regulator'](S_infer,
                                                            ylens=target_lengths,
                                                            n_quantizers=3,
                                                            f0=None)[0]
                cat_condition = torch.cat([prompt_condition, cond], dim=1)
                vc_target = self.s2mel.models['cfm'].inference(cat_condition,
                                                                target_lengths+prompt_condition.size(1),
                                                                ref_mel, style, None, diffusion_steps,
                                                                inference_cfg_rate=inference_cfg_rate)
                vc_target = vc_target[:, :, ref_mel.size(-1):]
                s2mel_time += time.perf_counter() - m_start_time
                wav_tmp = []
                m_start_time = time.perf_counter()
                wav = self.bigvgan(vc_target.float())
                audio_hop_len = self.cfg.s2mel['preprocess_params']['spect_params']['hop_length']
                wav_len = (target_lengths * audio_hop_len).long()
                
                # 只处理原始大小的段，舍弃填充的段
                for i in range(min(original_size, wav.size(0))):   
                    wav_i = wav[i:i+1,:,:wav_len[i]].squeeze().unsqueeze(0)
                    wav_i = torch.clamp(32767 * wav_i, -32767.0, 32767.0)
                    wav_tmp.append(wav_i)
                
                for wav in wav_tmp:
                    wavs.append(wav.cpu())
                bigvgan_time += time.perf_counter() - m_start_time

    end_time = time.perf_counter()

    self._set_gr_progress(0.9, "saving audio...")
    wavs = self.insert_interval_silence(wavs, sampling_rate=sampling_rate, interval_silence=interval_silence)
    wav = torch.cat(wavs, dim=1)
    wav_length = wav.shape[-1] / sampling_rate
    print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
    print(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
    print(f">> s2mel_time: {s2mel_time:.2f} seconds")
    print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
    print(f">> Total inference time: {end_time - start_time:.2f} seconds")
    print(f">> Generated audio length: {wav_length:.2f} seconds")
    print(f">> RTF: {(end_time - start_time) / wav_length:.4f}")

    # save audio
    wav = wav.cpu()  # to cpu
    if output_path:
        # 直接保存音频到指定路径中
        if os.path.isfile(output_path):
            os.remove(output_path)
            print(">> remove old wav file:", output_path)
        if os.path.dirname(output_path) != "":
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
        print(">> wav file saved to:", output_path)
        return output_path
    else:
        # 返回以符合Gradio的格式要求
        wav_data = wav.type(torch.int16)
        wav_data = wav_data.numpy().T
        return (sampling_rate, wav_data)

# 在原来 DiT 外面包了一层 padding 和图编译逻辑
def dit_forward_wapper(self, x, prompt_x, x_lens, t, style, cond, mask_content=False):
    if self.static:
        batch_size = x.size(0)
        T_max = self.max_seq_len
        T_actual = x.size(2)
        if T_actual > T_max:
            # 截断输入
            x = x[..., :T_max]
            prompt_x = prompt_x[..., :T_max]
            cond = cond[:, :T_max, :]
            # 同样需要钳制长度张量，否则掩码会出错
            x_lens = torch.clamp(x_lens, max=T_max)
        elif T_actual < T_max:
            # 填充输入
            pad_len = T_max - T_actual
            # x 和 prompt_x (B, C, T) -> 填充最后一个维度
            x = F.pad(x, (0, pad_len))
            prompt_x = F.pad(prompt_x, (0, pad_len))
            # cond (B, T, C) -> 填充倒数第二个维度
            cond = F.pad(cond, (0, 0, 0, pad_len))
        
        # 检查当前 batch_size 是否已有编译图，没有则创建
        if batch_size not in self.compiled_transformers:
            import torch_npu
            import torchair
            config = torchair.CompilerConfig()
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            self.compiled_transformers[batch_size] = torchair.inference.cache_compile(
                self.transformer.forward, dynamic=False, fullgraph=True, backend=npu_backend, ge_cache=True, cache_dir='./.torchair_cache/index-tts'+str(batch_size)
            )
        
        # 使用对应 batch_size 的编译图
        compiled_transformer = self.compiled_transformers[batch_size]
        return self.forward_original(x, prompt_x, x_lens, t, style, cond, mask_content, compiled_transformer)[..., :T_actual]
    else:
        self.compiled_transformers = {}
    return self.forward_original(x, prompt_x, x_lens, t, style, cond, mask_content, None)

# 修改 DiT 原来的 forward
# 1. mask padding 到最大长度以支持静态图
# 2. 一些 batch_size=1 输入填充到 x.shape[0]
def dit_forward(self, x, prompt_x, x_lens, t, style, cond, mask_content=False, compiled_transformer=None):
    """
        x (torch.Tensor): random noise
            shape: (batch_size, 80, T)
        prompt_x (torch.Tensor): reference mel + zero mel
            shape: (batch_size, 80, T)
        x_lens (torch.Tensor): mel frames output
            shape: (batch_size, 1)
        t (torch.Tensor): radshape: 
            shape: (batch_size, 2)    
        style (torch.Tensor): reference global style
            shape: (batch_size, 192)
        cond (torch.Tensor): semantic info of reference audio and altered audio
            shape: (batch_size, T, 512)
    
    """
    if style is not None and style.size(0) != x.size(0):  
        style = style.repeat(x.size(0)//style.size(0), 1)
    if t is not None and t.size(0) != x.size(0):
        t = t.repeat(x.size(0)//t.size(0))
    if x_lens is not None and x_lens.size(0) != x.size(0):
        x_lens = x_lens.repeat(x.size(0)//x_lens.size(0))

    class_dropout = False
    if self.training and torch.rand(1) < self.class_dropout_prob:
        class_dropout = True
    if not self.training and mask_content:
        class_dropout = True
    # cond_in_module = self.cond_embedder if self.content_type == 'discrete' else self.cond_projection
    cond_in_module = self.cond_projection

    B, _, T = x.size()


    t1 = self.t_embedder(t)  # (N, D) # t1 [2, 512]
    cond = cond_in_module(cond) # cond [2,1863,512]->[2,1863,512]

    x = x.transpose(1, 2) # [2,1863,80]
    prompt_x = prompt_x.transpose(1, 2) # [2,1863,80]

    x_in = torch.cat([x, prompt_x, cond], dim=-1) # 80+80+512=672 [2, 1863, 672]
    
    if self.transformer_style_condition and not self.style_as_token: # True and True
        x_in = torch.cat([x_in, style[:, None, :].repeat(1, T, 1)], dim=-1) #[2, 1863, 864]
        
    if class_dropout: #False
        x_in[..., self.in_channels:] = x_in[..., self.in_channels:] * 0 # 80维后全置为0
        
    x_in = self.cond_x_merge_linear(x_in)  # (N, T, D) [2, 1863, 512]
    
    if self.style_as_token: # False
        style = self.style_in(style)
        style = torch.zeros_like(style) if class_dropout else style
        x_in = torch.cat([style.unsqueeze(1), x_in], dim=1)
        
    if self.time_as_token: # False
        x_in = torch.cat([t1.unsqueeze(1), x_in], dim=1)
    
    if self.static:
        # --- 关键修改 ---
        # 1. 获取 x_in 的总序列长度 (T_max + 可能的
        T_seq = x_in.size(1) 
        
        # 2. 确保 x_lens 是 1D 张量 (B,) 以便广播
        # (x_lens 可能本是 (B, 1))
        mask_lengths = (x_lens + self.style_as_token + self.time_as_token).squeeze(-1) 
        
        # 3. 显式地将 T_seq 作为 max_length 传递给 sequence_mask
        # 假设 sequence_mask 函数已按上面示例定义
        x_mask = sequence_mask(mask_lengths, max_length=T_seq).to(x.device).unsqueeze(1) #torch.Size([B, 1, T_seq])
        # breakpoint()
        # --- 修改结束 ---
    else:    
        x_mask = sequence_mask(x_lens + self.style_as_token + self.time_as_token).to(x.device).unsqueeze(1) #torch.Size([1, 1, 1863])True
    # breakpoint()
    input_pos = self.input_pos[:x_in.size(1)]  # (T,) range（0，1863）
    # x_mask_expanded = x_mask[:, None, :].repeat(1, 1, x_in.size(1), 1) if not self.is_causal else None # torch.Size([1, 1, 1863, 1863]
    x_mask_expanded = x_mask.unsqueeze(-1) * x_mask.unsqueeze(-2)
    
    # 使用传入的 compiled_transformer 参数
    if compiled_transformer is None:
        x_res = self.transformer(x_in, t1.unsqueeze(1), input_pos, x_mask_expanded) # [2, 1863, 512]
    else:
        x_res = compiled_transformer(x_in, t1.unsqueeze(1), input_pos=input_pos, mask=x_mask_expanded, context=None, context_input_pos=None, cross_attention_mask=None) # [2, 1863, 512]
    x_res = x_res[:, 1:] if self.time_as_token else x_res
    x_res = x_res[:, 1:] if self.style_as_token else x_res
    
    if self.long_skip_connection: #True
        x_res = self.skip_linear(torch.cat([x_res, x], dim=-1))
    if self.final_layer_type == 'wavenet':
        x = self.conv1(x_res)
        x = x.transpose(1, 2)
        t2 = self.t_embedder2(t)
        x = self.wavenet(x, x_mask, g=t2.unsqueeze(2)).transpose(1, 2) + self.res_projection(
            x_res)  # long residual connection
        x = self.final_layer(x, t1).transpose(1, 2)
        x = self.conv2(x)
    else:
        x = self.final_mlp(x_res)
        x = x.transpose(1, 2)
    # x [2,80,1863]
    return x

def decode_with_embedding(self, input_ids=None,
                past_key_values=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                updated_kv_positions: Optional[torch.Tensor] = None,
                trunc_index=None):
    emb = self.embeddings(input_ids)
    emb = emb + self.text_pos_embedding.get_fixed_embedding_with_tensor_input(
        # attention_mask.shape[1] - mel_len, attention_mask.device 原来这一行和静态 cache 冲突了
        trunc_index
    )
    return self.transformer(
            inputs_embeds=emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            updated_kv_positions=updated_kv_positions,
        )

def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
    token_type_ids = kwargs.get("token_type_ids", None)  # usually None
    if not self.kv_cache:
        past_key_values = None
    # only last token for inputs_ids if past is defined in kwargs
    
    if past_key_values:
        input_ids = input_ids[:, -1].unsqueeze(-1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)
    if not self.static:
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
    updated_kv_positions = None
    if self.static:
        # 在本模型，decode 传入的 attention_mask 是固定的 [batch_size, seq_len] 全一, position_ids 是 None
        # 在改成静态 cache 后，kv 的 token 长度固定是 max_len=self.config.n_positions
        # attention_mask 形状也得改成 [batch_size, max_len]，并且每次 decode 往里面填 1 就行，不需要重新生成
        # 这个形状算子不认，需改成四维形式 [batch_size, 1, q_len=1, kv_len=max_len]
        batch_size, seq_length=input_ids.shape
        if past_key_values is None:
            if attention_mask is None:
                attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=input_ids.device)
            updated_kv_positions = torch.zeros(batch_size,dtype=input_ids.dtype,device=input_ids.device)
            position_ids=torch.arange(0, seq_length, dtype=torch.long,device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            self.last_decode_attention=None
            src_len = attention_mask.size(-1)
        else:
            bsz, src_len=attention_mask.size()
            if self.last_decode_attention is None:
                padding_mask=torch.zeros(batch_size,self.config.n_positions,device=input_ids.device)
                padding_mask[:,:src_len]=attention_mask
                attention_mask=padding_mask
                attention_mask=_prepare_decoder_attention_mask(attention_mask,(batch_size,seq_length),past_key_values[0][0], self.config.n_positions,)
                self.last_decode_attention = attention_mask
            else:
                attention_mask = self.last_decode_attention
                attention_mask[:,:,:,src_len:src_len+1]=0
            position_ids=torch.full((batch_size, 1), src_len, dtype=torch.long, device=attention_mask.device)
            updated_kv_positions=position_ids
        # 2.
        if past_key_values is None:
            kv_cache_type=torch.float16
            past_key_values = ()
            for _ in range(self.config.n_layer):
                kv_shape=(batch_size,
                    self.config.n_head // 1,
                    self.config.n_positions,
                    self.config.n_embd // self.config.n_head
                )
                k_cache=torch.zeros(kv_shape, dtype=kv_cache_type, device=input_ids.device)
                v_cache=torch.zeros(kv_shape, dtype=kv_cache_type, device=input_ids.device)
                past_key_values += ((k_cache, v_cache),)
            attention_mask=_prepare_decoder_attention_mask(attention_mask, (batch_size,seq_length),past_key_values[0][0], self.config.n_positions,)
    else:
        src_len = attention_mask.size(-1)
    # breakpoint()
    return {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "use_cache": kwargs.get("use_cache"),
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        'updated_kv_positions': updated_kv_positions,
        'real_seq_len': src_len,
    }

def gpt_forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        real_seq_len=None,
        updated_kv_positions: Optional[torch.Tensor] = None,
):
    assert self.cached_mel_emb is not None
    assert inputs_embeds is None  # Not supported by this inference model.
    assert labels is None  # Training not supported by this inference model.
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )
    # Create embedding
    mel_len = self.cached_mel_emb.shape[1]
    
    # 获取当前 batch_size
    batch_size = input_ids.shape[0]
    
    if input_ids.shape[1] != 1:
        text_inputs = input_ids[:, mel_len:]
        text_emb = self.embeddings(text_inputs)
        text_emb = text_emb + self.text_pos_embedding(text_emb)
        if self.cached_mel_emb.shape[0] != text_emb.shape[0]:
            mel_emb = self.cached_mel_emb.repeat_interleave(
                text_emb.shape[0] // self.cached_mel_emb.shape[0], 0
            )
        else:  # this outcome only occurs once per loop in most cases
            mel_emb = self.cached_mel_emb
        emb = torch.cat([mel_emb, text_emb], dim=1)
    elif self.compiled_transformers is None:
        emb = self.embeddings(input_ids)
        emb = emb + self.text_pos_embedding.get_fixed_embedding(
            attention_mask.shape[1] - mel_len, attention_mask.device
        )
    else:
        # 使用编译好的图
        pass
    
    # 如果是静态模式且输入长度为1，使用编译好的图
    if input_ids.shape[1] == 1 and self.compiled_transformers is not None:
        if batch_size not in self.compiled_transformers:
            config = torchair.CompilerConfig()
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            self.compiled_transformers[batch_size]=torchair.inference.cache_compile(self.decode_with_embedding, dynamic=not self.static, fullgraph=True, backend=npu_backend, ge_cache=True, cache_dir='./torchair_cache/index-tts'+str(batch_size))
        # 使用对应 batch_size 的编译图
        compiled_transformer = self.compiled_transformers[batch_size]
        # breakpoint()
        transformer_outputs = compiled_transformer(
            input_ids=input_ids.contiguous(), # 防止重编译
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            updated_kv_positions=updated_kv_positions,
            trunc_index=torch.tensor(real_seq_len+1-mel_len,device=input_ids.device),
        )
    else:
        # 常规执行路径
        transformer_outputs = self.transformer(
            inputs_embeds=emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            updated_kv_positions=updated_kv_positions,
        )
    
    # breakpoint()
    hidden_states = transformer_outputs[0]

    # Set device for model parallelism
    if self.model_parallel:
        if torch.backends.mps.is_available():
            self.to(self.transformer.first_device)
        else:
            torch.cuda.set_device(self.transformer.first_device)
        hidden_states = hidden_states.to(self.lm_head.weight.device)

    lm_logits = self.lm_head(hidden_states)

    if not return_dict:
        return (lm_logits,) + transformer_outputs[1:]

    return CausalLMOutputWithCrossAttentions(
        loss=None,
        logits=lm_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
        cross_attentions=transformer_outputs.cross_attentions,
    )

def get_fixed_embedding_with_tensor_input(self, ind):
    return self.emb(ind).unsqueeze(0)

def dit_attn_forward(self,
            x: Tensor,
            freqs_cis: Tensor,
            mask: Tensor,
            input_pos: Optional[Tensor] = None,
            context: Optional[Tensor] = None,
            context_freqs_cis: Optional[Tensor] = None,
            ) -> Tensor:
    bsz, seqlen, _ = x.shape

    kv_size = self.n_local_heads * self.head_dim
    if context is None:
        q, k, v = self.wqkv(x).split([kv_size, kv_size, kv_size], dim=-1)
        context_seqlen = seqlen
    else:
        q = self.wq(x)
        k, v = self.wkv(context).split([kv_size, kv_size], dim=-1)
        context_seqlen = context.shape[1]

    q = q.view(bsz, seqlen, self.n_head, self.head_dim)
    k = k.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)
    v = v.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)

    q = apply_rotary_emb(q, freqs_cis)
    k = apply_rotary_emb(k, context_freqs_cis if context_freqs_cis is not None else freqs_cis)

    q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

    if self.kv_cache is not None:
        k, v = self.kv_cache.update(input_pos, k, v)

    k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
    v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
    # breakpoint()
    _, N, _, D = q.size()
    q=q.contiguous().half()
    k=k.contiguous().half()
    v=v.contiguous().half()
    scale=1.0 / (D**0.5)
    # y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
    y, _ = torch_npu.npu_fused_infer_attention_score(
        q, k, v,
        atten_mask=mask.logical_not(),
        num_heads=N,
        scale=scale,
        input_layout='BNSD'
    )

    y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.head_dim * self.n_head)

    y = self.wo(y)
    return y

def gpt_attn_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = False,
    updated_kv_positions: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
    is_cross_attention = encoder_hidden_states is not None
    if is_cross_attention:
        if not hasattr(self, "q_attn"):
            raise ValueError(
                "If class is used as cross attention, the weights `q_attn` have to be defined. "
                "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
            )

        query_states = self.q_attn(hidden_states)
        key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
        attention_mask = encoder_attention_mask
    else:
        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2) # B,S,N*D=H

    shape_q = (*query_states.shape[:-1], -1, self.head_dim)
    shape_kv = (*key_states.shape[:-1], -1, self.head_dim)

    query_states = query_states.view(shape_q).transpose(1, 2) # B,N,S,D
    key_states = key_states.view(shape_kv).transpose(1, 2)
    value_states = value_states.view(shape_kv).transpose(1, 2)

    if past_key_value is not None and not self.static:
        if isinstance(past_key_value, EncoderDecoderCache):
            if is_cross_attention:
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache
        cache_kwargs = {"cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs=cache_kwargs
        )

    is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention

    using_eager = self.config._attn_implementation == "eager"
    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and (output_attentions or head_mask is not None):
            using_eager = True
            # logger.warning_once(
            #     "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
            #     'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            # )
        else:
            # Attention functions are consistent with previous equivalent attention classes, however they do not support some options
            # (e.g. layer scaling, head mask) that eager supports. These implementations are thus equivalent to previous code, but
            # not necessarily to eager (if mentioned options are provided).
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    if using_eager and self.reorder_and_upcast_attn:
        attn_output, attn_weights = self._upcast_and_reordered_attn(
            query_states, key_states, value_states, attention_mask, head_mask
        )
    else:
        if self.static:
            if attention_mask is not None:
                attention_mask = attention_mask.bool()
            head_num = query_states.shape[1]
            if updated_kv_positions is not None:
                q_len=query_states.shape[-2]
                tmp_ids=updated_kv_positions.reshape(-1)
                torch_npu.scatter_update_(past_key_value[self.layer_idx][0], tmp_ids, key_states, -2)
                torch_npu.scatter_update_(past_key_value[self.layer_idx][1], tmp_ids, value_states, -2)
                key_states = past_key_value[self.layer_idx][0] if q_len == 1 else key_states
                value_states = past_key_value[self.layer_idx][1] if q_len == 1 else value_states
            if updated_kv_positions is not None:
                attn_output, attn_weights = torch_npu.npu_fused_infer_attention_score(query_states,
                    key_states, value_states, num_heads=head_num, input_layout='BNSD',atten_mask=attention_mask,
                    scale=1.0/math.sqrt(query_states.shape[-1]))
                attn_output = attn_output.transpose(2, 1).contiguous()
            else:
                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    head_mask=head_mask,
                    dropout=self.attn_dropout.p if self.training else 0.0,
                    is_causal=is_causal,
                    **kwargs,
                )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                head_mask=head_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
                **kwargs,
            )
    attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    return attn_output, attn_weights

# 函数替换
IndexTTS2.infer_fast = infer_fast
IndexTTS2.make_batch = make_batch
GPT2InferenceModel.decode_with_embedding = decode_with_embedding
GPT2InferenceModel.prepare_inputs_for_generation = prepare_inputs_for_generation
GPT2InferenceModel.forward = gpt_forward
LearnedPositionEmbeddings.get_fixed_embedding_with_tensor_input = get_fixed_embedding_with_tensor_input
DiT.forward_original = dit_forward
DiT.forward = dit_forward_wapper
Attention.forward = dit_attn_forward
GPT2Attention.forward = gpt_attn_forward

if __name__ == "__main__":
    import os
    enable_prof = os.getenv('enable_prof')
    if enable_prof:
        enable_prof = enable_prof.lower() == 'true'
    if enable_prof:
        import torch_npu
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=[
                torch_npu.profiler.ExportType.Text,
                torch_npu.profiler.ExportType.Db
                ],
            profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
            msprof_tx=False,
            mstx_domain_include=[],
            mstx_domain_exclude=[],
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            l2_cache=False,
            op_attr=False,
            data_simplification=False,
            record_op_args=False,
            gc_detect_threshold=None,
            host_sys=[
                torch_npu.profiler.HostSystem.CPU,
                torch_npu.profiler.HostSystem.MEM],
            sys_io=False,
            sys_interconnection=False
        )
        with_stack=os.getenv('with_stack')
        if with_stack is not None:
            with_stack = with_stack.lower() == 'true'
        prof = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU
                ],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
            record_shapes=False,
            profile_memory=False,
            with_stack=with_stack,
            with_modules=False,
            with_flops=False,
            experimental_config=experimental_config)
    tts = IndexTTS2NPU(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_cuda_kernel=True, use_fp16=True, static=True)
    # breakpoint()
    prompt_wav = "examples/voice_01.wav"
    batch_size=16
    text = 'a' * batch_size # warmup
    # text= '春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。风急天高猿啸哀，渚清沙白鸟飞回。无边落木萧萧下，不尽长江滚滚来。万里悲秋常作客，百年多病独登台。艰难苦恨繁霜鬓，潦倒新停浊酒杯。唧唧复唧唧，木兰当户织。不闻机杼声，惟闻女叹息。脱我战时袍，著我旧时裳。当窗理云鬓，对镜帖花黄。明月几时有？把酒问青天。不知天上宫阙，今夕是何年。我欲乘风归去，又恐琼楼玉宇，高处不胜寒。起舞弄清影，何似在人间。但愿人长久，千里共婵娟。' * 4
    tts.infer_fast(spk_audio_prompt=prompt_wav, text=text, output_path="gen.wav", verbose=False, max_text_tokens_per_segment=1, num_beams=1, auto_split=False)
    if enable_prof: 
        prof.start()
        text='测试，测试，测试，测试，测试'
        tts.infer_fast(spk_audio_prompt=prompt_wav, text=text, output_path="gen2.wav", verbose=False, max_text_tokens_per_segment=6, num_beams=1, auto_split=False)
    else:
        prompt_wav = "examples/voice_01.wav"
        for i in range(1):
            # text = '短句。'
            # text = '春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。风急天高猿啸哀，渚清沙白鸟飞回。无边落木萧萧下，不尽长江滚滚来。万里悲秋常作客，百年多病独登台。艰难苦恨繁霜鬓，潦倒新停浊酒杯。唧唧复唧唧，木兰当户织。不闻机杼声，惟闻女叹息。脱我战时袍，著我旧时裳。当窗理云鬓，对镜帖花黄。明月几时有？把酒问青天。不知天上宫阙，今夕是何年。我欲乘风归去，又恐琼楼玉宇，高处不胜寒。起舞弄清影，何似在人间。但愿人长久，千里共婵娟。千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。故人西辞黄鹤楼，烟花三月下扬州。孤帆远影碧空尽，唯见长江天际流。'
            text= '春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。风急天高猿啸哀，渚清沙白鸟飞回。无边落木萧萧下，不尽长江滚滚来。万里悲秋常作客，百年多病独登台。艰难苦恨繁霜鬓，潦倒新停浊酒杯。唧唧复唧唧，木兰当户织。不闻机杼声，惟闻女叹息。脱我战时袍，著我旧时裳。当窗理云鬓，对镜帖花黄。明月几时有？把酒问青天。不知天上宫阙，今夕是何年。我欲乘风归去，又恐琼楼玉宇，高处不胜寒。起舞弄清影，何似在人间。但愿人长久，千里共婵娟。' * 4
            tts.infer_fast(spk_audio_prompt=prompt_wav, text=text, output_path="gen2.wav", verbose=False, max_text_tokens_per_segment=120, num_beams=1, auto_split=False)
    if enable_prof:
        prof.step()
        prof.stop()
