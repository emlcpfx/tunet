Looking at your loss curve and logs:

You're at ~step 2160 (epoch 3, 160/1000 into epoch 3)
Best checkpoint was already saved around epoch 1.4 (the stars)
The curve has largely plateaued — training loss is hovering ~0.015-0.016, not trending meaningfully down anymore
Short answer: you can stop now, or let it finish epoch 3 (~2840 more steps). Use the best checkpoint, not the final one.

For a 2-pair flicker fix specifically:

You want moderate overfitting to those frames — that's the point
Best was 0.00590 at ~epoch 1.4, which is your checkpoint to use
Running past ~3000-4000 total steps risks the model overfitting too hard and breaking generalization on surrounding frames
The dotted validation line has also flattened, confirming diminishing returns
Recommendation: Stop at end of epoch 3 (step ~3000), load the best checkpoint, and test it. If flickering persists, you may need more pairs rather than more steps.