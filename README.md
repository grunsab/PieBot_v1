
# A chess engine based on the AlphaZero algorithm

This is a pytorch implementation of Google Deep Mind's AlphaZero algorithm for chess.

## Live

A LiChess Bot is available at the following link. The bot allows anyone to play with the current iteration of the chess engine.

https://lichess.org/@/PieBot

Watch it play live games against other bots here:
https://lichess.org/@/PieBot/tv

## Dependencies

Standard python libraries. See requirements.txt.

## Running the chess engine

The entry point to the chess engine is the python file playchess.py. Good parameters for strong, long-thinking moves would be:
```
python3 playchess.py --model PieBot_20x256_v0.pt --verbose --rollouts 500 --threads 20 --mode h
```
The current position is displayed with an ascii chess board. Enter your moves in long algebraic notation. Note that running the engine requires a weights file.  


## Training script

Download the 2.5MM games [CCRL Dataset](https://lczero.org/blog/2018/09/a-standard-dataset/), reformat it using `reformat.py`and run `train.py`.

Use download_computerchess_org_uk.py program to download another 2MM games. Note that the LCZero Standard Dataset above of 2.5MM is a subset of computerchess.org.uk, which has a total of 3.83 million games. 

Use download_lichess_games.py to download more 3000+ ELO games. Note that there are approximately 400k such games spread across all LiChess's many years of operating, so it might not be that efficient to use download_lichess_games.py.

For tips on how to use reformat.py, reference the bottom of the reformat.py, which has all the arguments that reformat offers. It should take no more than 60 minutes to download the CCRL Dataset, and reformat all th files using reformat on a modern system. You may need to decrease the number of threads used by reformat.py to a sane number (100 for example) to avoid file access issues, if you are using a processor with a very high number of cores.

For train.py, if you have access to an NVIDIA GPU, than you can leave the settings as is. You should have a fully trained chess engine that plays at 2700 ELO in about seven days on a consumer GPU like a 3090 or 4080. If you don't have access to a high end GPU, you might want to decrease the size of the model that's being trained for, by decreasing the number of blocks and filters. You might want to decrease the number of epochs to 40 or 50, since you'll notice diminishing returns as number of epochs increases.

You can run 

```
python3 train.py --mode supervised --epochs 20 --lr 0.0005
```

 to train the supervised algorithm once you have all the games downloaded and reformatted.
 


 To train using supervised learning first, and then use reinforcement learning on the model that results from supervised learning, you can use 

 ```
 python3 train_curriculum.py
 ```

Check the bottom of train_curriculum.py to see how to change the settings.


## About the algorithm

The algorithm is based on [this paper](https://arxiv.org/pdf/1712.01815.pdf). One very important difference between the algorithm used here and the one described in that paper is that this implementation used supervised learning and optionally reinforcement learning, instead of solely reinforcement learning. Doing reienforcement learning is very computationally intensive. As said in that paper, it took thousands of TPUs to generate the self play games. This program, on the other hand, starts supervised training on the [CCRL Dataset](https://lczero.org/blog/2018/09/a-standard-dataset/), which contains 2.5 million top notch chess games. Because each game has around 80 unique positions in it, this yields about 200 million data points for training on. If you want you can augment that with download_computerchess_org_uk.py for another approximately 1MM games. If you want to use reinforcement learning after that, that's possible and should further strengthen the engine.

## Strength

Note that there are multiple models included in the repository. PieBot_20x256_v0.pt is currently the most refined of them, with a policy loss of approximately 1.5 and a value loss of approximately 0.4. It was trained across 250 epochs on a dataset of 2MM games from LCZero's standard dataset, using supervised learning.

The current best model performs at around 2400 ELO on LiChess (available to test on PieBot_20x256_v0.pt), on a Macbook Mini M4 at 400-500 nodes per second evaluated. It performs at a higher ELO of around 2500 on a Macbook Pro M4 Pro due to that device processing 800 nodes per second.

I'm training a new model using a larger dataset including the ones from ComputerChess.org.uk, which should play much stronger once it's completed. 

I experimented with increasing the number of positions evaluated by adjusting the MCTS, but I was not sucessful at improving the overall throughput of the model in terms of nodes per second evaluated (see the folder experiments). I was surprised to see that Google DeepMind claims a performance of 80k positions evaluated per second on a 4 TPU setup. I did however create a benchmark_nn.py script, which shows that the maximum number of nodes per second that my Macbook Pro M4 Pro can reach is 1600 nodes per second, and the maximum number of nodes per second that my RTX 4080 can reach is 3600 nodes per second. I reach approximately 1000 nodes per second on each device right now, which suggests a significant bottleneck on the MCTS code.

I am however now experimenting with model quantization, that should hopefully increase the nodes evaluated per second significantly, if I can somehow get the quant version to run on GPU, which PyTorch doesn't offically support.

