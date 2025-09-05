# Parameter-Efficient Adapter Integration for Echocardiographic Video Segmentation


</div>

The increasing demand for echocardiographic examinations contrasts with the shortage of specialized medical professionals. To address this challenge, we propose a parameter-efficient adaptation of SAMWISE for automated cardiac video segmentation through systematic adapter integration.
Our approach integrates adapter modules into the SAMWISE framework, enabling effective domain adaptation for consistent temporal segmentation of cardiac structures. Specifically, we present a method that achieves substantial segmentation improvements using adapters with minimal trainable parameters, even in environments where training large-scale backbone models with limited medical data is challenging. This parameter-efficient strategy makes our approach particularly suitable for practical clinical deployment. 


## Citation

```
@InProceedings{cuttano2025samwise,
    author    = {Cuttano, Claudia and Trivigno, Gabriele and Rosi, Gabriele and Masone, Carlo and Averta, Giuseppe},
    title     = {SAMWISE: Infusing Wisdom in SAM2 for Text-Driven Video Segmentation},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {3395-3405}
}
```

```
Tejero, J. G., Schmid, M., Neila, P. M., Zinkernagel, M. S., Wolf, S., & Sznitman, R. (2025, February). SAM-DA: Decoder Adapter for Efficient Medical Domain Adaptation. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) (pp. 6775-6784). IEEE.
```

```
S. Leclerc, E. Smistad, J. Pedrosa, A. Ostvik, F. Cervenansky, F. Espinosa, T. Espeland, E. A. R. Berg, P.-M. Jodoin, T. Grenier, C. Lartizien, J. Dhooge, L. Løvstakken, and O. Bernard, "Deep Learning for Segmentation using an Open Large-Scale Dataset in 2D Echocardiography," IEEE Transactions on Medical Imaging, vol. 38, no. 9, pp. 2198-2210, 2019.
```

## Dataset
This work uses the EchoNet-Dynamic dataset:
- Ouyang, D., He, B., Ghorbani, A. et al. Video-based AI for beat-to-beat assessment of cardiac function. Nature 580, 252–256 (2020). https://doi.org/10.1038/s41586-020-2145-8
- Dataset DOI: https://doi.org/10.71718/yqp5-y078

This work uses the CAMUS (Cardiac Acquisitions for Multi-structure Ultrasound Segmentation) dataset:
- 500 patients with 2D apical four-chamber and two-chamber views
- Fully annotated for cardiac structure segmentation
- Available at: https://www.creatis.insa-lyon.fr/Challenge/camus/

## Data Usage Agreement
The EchoNet-Dynamic dataset is used for non-commercial research purposes only, in accordance with the Stanford University School of Medicine data usage agreement.