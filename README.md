# oneAPI_Hackathon

JamesZhutheThird

## Updates
- 20230826 v1.1 Add customized FFT (however there are bugs)
- 20230825 v1.0 Support FFT with `oneMKL` and `fftw3`

## Performance

|   Matrix M | Matrix N |  customized  |  fftw3 (acc)  |  oneMKL (acc)  |
|-----------:|:---------|:------------:|:-------------:|:--------------:|
|          8 | 8        |      -       |       -       |       -        |
|         16 | 16       |      -       |       -       |       -        |
|         64 | 64       |      -       |       -       |       -        |
|        256 | 256      |      -       |       -       |       -        |
|       1024 | 1024     |      -       |       -       |       -        |
|          1 | 4096     |      -       |       -       |       -        |
|         64 | 4096     |      -       |       -       |       -        |
|       4096 | 4096     |      -       |       -       |       -        |
|          1 | 16384    |      -       |       -       |       -        |
|        128 | 16384    |      -       |       -       |       -        |
|      16384 | 16384    |      -       |       -       |       -        |
|          1 | 65536    |      -       |       -       |       -        |
|        512 | 65536    |      -       |       -       |       -        |
|      65536 | 65536    |      -       |       -       |       -        |