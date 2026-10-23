# T02 development benchmark

Development synthetic load; official acceptance is false.

Requested input lengths describe generated text. Image/chat tokens and detokenization mismatch require server-side calibration.

| Round | Text target | Output target | Concurrency | Expected | Completed | Failed | Native duration s | Request/s | Output token/s | Mean TTFT ms | p99 TTFT ms | Mean TPOT ms | p99 TPOT ms | Mean E2EL ms | p99 E2EL ms | Client wall s | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16384 | 1024 | 64 | 128 | 128 | 0 | 1335.1635527042672 | 0.09586840484129919 | 98.16924655749037 | 407773.81381296436 | 600746.0267734993 | 147.8136000478934 | 155.29854754821278 | 558987.1266619593 | 757967.3596001789 | 1388.66132 | complete |
| 2 | 16384 | 1024 | 64 | 128 |  |  |  |  |  |  |  |  |  |  |  | 1307.366197 | timeout |
