# Robustify Spiking Neural Networks via Dominant Singular Deflation under Heterogeneous Training Vulnerability (<a href="https://openreview.net/forum?id=EIYltBaUzL">ICLR 2026</a>)


**Desong Zhang, Jia Hu, Geyong Min**  
  University of Exeter  

### Quick Start with
``` 
python train.py -model 'vgg11' -dataset 'cifar10'
```
### Configuration

Hyperparameters and dataset settings can be configured in `./utils/config.py`

### Paper Abstract

Spiking Neural Networks (SNNs) process information via discrete spikes, enabling them to operate at remarkably low energy levels. However, our experimental observations reveal a striking vulnerability when SNNs are trained using the mainstream method—direct encoding combined with backpropagation through time (BPTT): even a single backward pass on data drawn from a slightly different distribution can lead to catastrophic network collapse. We refer to this phenomenon as the heterogeneous training vulnerability of SNNs. Our theoretical analysis attributes this vulnerability to the repeated inputs inherent in direct encoding and the gradient accumulation characteristic of BPTT, which together produce an exceptional large Hessian spectral radius. To address this challenge, we develop a hyperparameter-free method called Dominant Singular Deflation (DSD). By orthogonally projecting the dominant singular components of gradients, DSD effectively reduces the Hessian spectral radius, thereby preventing SNNs from settling into sharp minima. Extensive experiments demonstrate that DSD not only mitigates the vulnerability of SNNs under heterogeneous training, but also significantly enhances overall robustness compared to key baselines, providing strong support for safer SNNs.

### Acknowledgements

This implementation is based on the <a href="https://github.com/fangwei123456/spikingjelly">SpikingJelly</a> framework. We sincerely thank the authors for making their code publicly available. 



