import numpy as np
import matplotlib.pyplot as plt

epo=300

loss_fname = './T4/loss/4c_loss_history.txt'
numpy_loss_history = np.loadtxt(loss_fname)

val_loss_fname = './T4/loss/4c_val_loss_history.txt'
numpy_val_loss_history = np.loadtxt(val_loss_fname)

pred_mask_fname = './T4/pred_masks/4c_pred_mask.txt'
numpy_p_m = np.loadtxt(pred_mask_fname)

plt.figure(0)
plt.plot(numpy_val_loss_history,numpy_loss_history, 'bo')
plt.xlabel('validation loss')
plt.ylabel('training loss')

plt.figure(1)
plt.plot(np.linspace(1,epo,num=epo),numpy_loss_history)
plt.ylabel('loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(np.linspace(1,epo,num=epo),numpy_val_loss_history)
plt.ylabel('validation loss')
plt.xlabel('epoch')

plt.figure(3)
plt.imshow(numpy_p_m)
plt.show()