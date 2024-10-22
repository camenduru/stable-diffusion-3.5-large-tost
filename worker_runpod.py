import os, json, requests, runpod

import random, time
import torch
from PIL import Image
import numpy as np

import nodes
from nodes import NODE_CLASS_MAPPINGS
from comfy_extras import nodes_sd3

TripleCLIPLoader = nodes_sd3.NODE_CLASS_MAPPINGS["TripleCLIPLoader"]()
EmptySD3LatentImage = nodes_sd3.NODE_CLASS_MAPPINGS["EmptySD3LatentImage"]()
CheckpointLoaderSimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

with torch.inference_mode():
    clip = TripleCLIPLoader.load_clip("clip_l.safetensors", "clip_g.safetensors", "t5xxl_fp16.safetensors")[0]
    unet, _, vae = CheckpointLoaderSimple.load_checkpoint("sd3.5_large.safetensors")

def closestNumber(n, m):
    q = int(n / m)
    n1 = m * q
    if (n * m) > 0:
        n2 = m * (q + 1)
    else:
        n2 = m * (q - 1)
    if abs(n - n1) < abs(n - n2):
        return n1
    return n2

@torch.inference_mode()
def generate(input):
    values = input["input"]

    positive_prompt = values['positive_prompt']
    negative_prompt = values['negative_prompt']
    width = values['width']
    height = values['height']
    seed = values['seed']
    steps = values['steps']
    cfg = values['cfg']
    sampler_name = values['sampler_name']
    scheduler = values['scheduler']

    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)
    print(seed)

    cond = nodes.CLIPTextEncode().encode(clip, positive_prompt)[0]
    n_cond = nodes.CLIPTextEncode().encode(clip, negative_prompt)[0]
    empty_latent_image=EmptySD3LatentImage.generate(width, height, batch_size=1)[0]
    sample = nodes.common_ksampler(model=unet,
                                    seed=seed,
                                    steps=steps,
                                    cfg=cfg,
                                    sampler_name=sampler_name,
                                    scheduler=scheduler,
                                    positive=cond,
                                    negative=n_cond,
                                    latent=empty_latent_image,
                                    denoise=1.0)[0]
    decoded = VAEDecode.decode(vae, sample)[0].detach()
    Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0]).save("/content/stable-diffusion-3.5-large-tost.png")

    result = "/content/stable-diffusion-3.5-large-tost.png"
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)

runpod.serverless.start({"handler": generate})