import argparse
import os
import json
import numpy as np
import torch

from lpmc.music_captioning.model.bart import BartCaptionModel
from lpmc.utils.eval_utils import load_pretrained
from lpmc.utils.audio_utils import load_audio, STR_CH_FIRST
from omegaconf import OmegaConf


parser = argparse.ArgumentParser(description='Music Captioning Batch')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument("--framework", default="transfer", type=str)
parser.add_argument("--caption_type", default="lp_music_caps", type=str)
parser.add_argument("--num_beams", default=5, type=int)
parser.add_argument("--model_type", default="last", type=str)

# 🔥 NEW
parser.add_argument("--audio_dir", type=str, required=True)
parser.add_argument("--output_json", default="results.json", type=str)


def get_audio(audio_path, duration=10, target_sr=16000):
    n_samples = int(duration * target_sr)
    audio, sr = load_audio(
        path=audio_path,
        ch_format=STR_CH_FIRST,
        sample_rate=target_sr,
        downmix_to_mono=True,
    )

    if len(audio.shape) == 2:
        audio = audio.mean(0, False)

    if audio.shape[-1] < n_samples:
        pad = np.zeros(n_samples)
        pad[: audio.shape[-1]] = audio
        audio = pad

    ceil = int(audio.shape[-1] // n_samples)
    audio = torch.from_numpy(
        np.stack(np.split(audio[:ceil * n_samples], ceil)).astype('float32')
    )

    return audio


def load_model(args):
    save_dir = f"exp/{args.framework}/{args.caption_type}/"
    config = OmegaConf.load(os.path.join(save_dir, "hparams.yaml"))

    model = BartCaptionModel(max_length=config.max_length)

    model, _ = load_pretrained(
        args,
        save_dir,
        model,
        model_types=args.model_type,
        mdp=config.multiprocessing_distributed
    )

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    model.eval()

    return model


def caption_file(model, audio_path, args):
    audio_tensor = get_audio(audio_path)

    audio_tensor = audio_tensor.cuda(args.gpu, non_blocking=True)

    with torch.no_grad():
        output = model.generate(
            samples=audio_tensor,
            num_beams=args.num_beams,
        )

    results = []
    for chunk, text in enumerate(output):
        time = f"{chunk * 10}:00-{(chunk + 1) * 10}:00"
        results.append({
            "time": time,
            "text": text
        })

    return results


def main():
    args = parser.parse_args()

    model = load_model(args)

    all_results = {}

    audio_files = [
        f for f in os.listdir(args.audio_dir)
        if f.endswith(".wav")
    ]

    print(f"Found {len(audio_files)} files")

    for file in audio_files:
        path = os.path.join(args.audio_dir, file)

        print(f"\nProcessing: {file}")

        try:
            captions = caption_file(model, path, args)
            all_results[file] = captions

        except Exception as e:
            print(f"Error with {file}: {e}")

    # 🔥 save JSON
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
