# SpinTRAK

## Environment set up

### Prerequisites
* Python version `3.12`
* CUDA 12.8

### Set up
To set up the environment run the following commands:
* `uv venv --python 3.12`
* `source venv/bin/activate`
* `uv pip install -e .`

## Running single generation
`spintrak generate-audio --prompt "A calm piano melody" --duration 30`

## Running single attribution scores calculation
`spintrak generate-gradients --prompt "calm ocean waves" --duration 30 --output ./calm_ocean_gradients.pt`

## Running experiments

To run experiments please refer to the [Experiments.md](./Experiments.md) file in the `experiments` directory.

### Endpoint details

`post : http://209.20.159.8:8000/gradient`

req body: {
  "generated_song_url": "https://mgaitesting.s3.us-east-1.amazonaws.com/Blues/Blues-1.wav",
  "prompt": "sad blues soulful",
  "duration": 30.0
}

 response body:
{
    "message": "Audio processed successfully",
    "genre": "f8f265ca_7",
    "metadata": {
        "model_used": "MusicGen",
        "finetune_checkpoint": "/home/ubuntu/musicgen_finetunes/f8f265ca_7",
        "training_dataset_size": 99,
        "gradient_dataset_size": 99,
        "time_per_second_audio": 0,
        "time_per_second_training_gradients": 0,
        "time_per_second_generated_gradients": 0,
        "gpu_model": "NVIDIA H100 PCIe",
        "prompt": "sad blues soulful",
        "seed": 42,
        "title": "f8f265ca_7_temp_1764232136_sad_blues_soulful_30.pt",
        "timestamp": "2025-11-27 08:33:42",
        "audio_length_seconds": 30.0
    },
    "top_similar_tracks": [
        {
            "rank": 1,
            "similarity_score": 750.0,
            "sample_index": 93,
            "track": "Cabaret Voltaire - If the Shadows Could March_instrumental.wav",
            "percentage": 35.83226013183594,
            "direction": "positively"
        },
        {
            "rank": 2,
            "similarity_score": 209.03515625,
            "sample_index": 0,
            "track": "$uicideboy$, Denzel Curry - Ultimate $uicide_instrumental.wav",
            "percentage": 9.986936569213867,
            "direction": "positively"
        },
        {
            "rank": 3,
            "similarity_score": 203.955078125,
            "sample_index": 57,
            "track": "Banco De Gaia - Last Train to Lhasa [High quality]_instrumental.wav",
            "percentage": 9.74422836303711,
            "direction": "positively"
        },
        {
            "rank": 4,
            "similarity_score": 195.849609375,
            "sample_index": 91,
            "track": "Byetone - Plastic Star - Session [High quality]_instrumental.wav",
            "percentage": 9.356978416442871,
            "direction": "positively"
        },
        {
            "rank": 5,
            "similarity_score": 137.064453125,
            "sample_index": 79,
            "track": "Big Walter Horton - Easy - Remastered 2022_instrumental.wav",
            "percentage": 6.548439025878906,
            "direction": "positively"
        },
        {
            "rank": 6,
            "similarity_score": 119.5703125,
            "sample_index": 9,
            "track": "8Ball & MJG - Space Age Pimpin__instrumental.wav",
            "percentage": 5.712632656097412,
            "direction": "positively"
        },
        {
            "rank": 7,
            "similarity_score": 104.005859375,
            "sample_index": 55,
            "track": "Baauer - Harlem Shake_instrumental.wav",
            "percentage": 4.969019889831543,
            "direction": "positively"
        },
        {
            "rank": 8,
            "similarity_score": 90.591796875,
            "sample_index": 86,
            "track": "Boards of Canada - Roygbiv_instrumental.wav",
            "percentage": 4.3281450271606445,
            "direction": "positively"
        },
        {
            "rank": 9,
            "similarity_score": 69.376953125,
            "sample_index": 47,
            "track": "Artful Dodger, Craig David - Re-Rewind (The Crowd Say Bo Selecta) (feat. Craig David) - Radio Edit_instrumental.wav",
            "percentage": 3.314577341079712,
            "direction": "positively"
        },
        {
            "rank": 10,
            "similarity_score": 66.462890625,
            "sample_index": 43,
            "track": "Arcade Fire - Wake Up_instrumental.wav",
            "percentage": 3.175354242324829,
            "direction": "positively"
        },
        {
            "rank": 11,
            "similarity_score": 41.041015625,
            "sample_index": 95,
            "track": "Carl Craig - Landcruising_instrumental.wav",
            "percentage": 1.9607897996902466,
            "direction": "positively"
        },
        {
            "rank": 12,
            "similarity_score": 29.31640625,
            "sample_index": 11,
            "track": "A Tribe Called Quest - Can I Kick It - J. Cole Remix_instrumental.wav",
            "percentage": 1.4006308317184448,
            "direction": "positively"
        },
        {
            "rank": 13,
            "similarity_score": 20.5234375,
            "sample_index": 32,
            "track": "Alvin Lee - The Bluest Blues_instrumental.wav",
            "percentage": 0.9805348515510559,
            "direction": "positively"
        },
        {
            "rank": 14,
            "similarity_score": 17.0859375,
            "sample_index": 76,
            "track": "Big Maceo - Worried Life Blues_instrumental.wav",
            "percentage": 0.8163036704063416,
            "direction": "positively"
        },
        {
            "rank": 15,
            "similarity_score": 16.982421875,
            "sample_index": 89,
            "track": "Borislav Slavov - Flooded Streets  Aquarium [High quality]_instrumental.wav",
            "percentage": 0.8113580942153931,
            "direction": "positively"
        },
        {
            "rank": 16,
            "similarity_score": 8.7421875,
            "sample_index": 65,
            "track": "Bentley Rhythm Ace - Bentley_s Gonna Sort You Out_instrumental.wav",
            "percentage": 0.41766977310180664,
            "direction": "positively"
        },
        {
            "rank": 17,
            "similarity_score": 8.615234375,
            "sample_index": 56,
            "track": "Bahamadia - Uknowhowwedu_instrumental.wav",
            "percentage": 0.4116044342517853,
            "direction": "positively"
        },
        {
            "rank": 18,
            "similarity_score": 4.37109375,
            "sample_index": 52,
            "track": "Autechre - Gantz Graf [Chosen by fans on Warp20.net]_instrumental.wav",
            "percentage": 0.20883488655090332,
            "direction": "positively"
        },
        {
            "rank": 19,
            "similarity_score": 0.49609375,
            "sample_index": 15,
            "track": "Above The Law - Murder Rap_instrumental.wav",
            "percentage": 0.02370154671370983,
            "direction": "positively"
        }
    ]
}
