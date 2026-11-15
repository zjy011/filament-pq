# Third-party code

`filament/unet/cldice.py` adapts the 2D soft skeleton and clDice loss from
[jocpae/clDice](https://github.com/jocpae/clDice/tree/master/cldice_loss/pytorch),
accessed 2026-11-15. It uses ten erosion iterations, unit smoothing, batch-pooled
foreground sums, and a single foreground channel. The project objective adds
0.5 times this loss to Dice + BCE; it does not use the repository's mixed-Dice
objective. See Shit et al., [clDice, CVPR 2021](https://arxiv.org/abs/2003.07311).

MIT License

Copyright (c) 2021 Johannes C. Paetzold and Suprosanna Shit

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
