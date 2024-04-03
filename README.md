# IHSetHansonKraus1991
Python package to run and calibrate Hanson and Kraus (1991) one-line model.

## :house: Local installation
* Using pip:
```bash

pip install git+https://github.com/defreitasL/IHSetHansonKraus1991.git

```

---
## :zap: Main methods

* [HansonKraus1991](./IHSetHansonKraus1991/HansonKraus1991.py):
```python
# model's it self
hansonKraus1991(yi, dt, dx, hs, tp, dir, depth, doc, kal, X0, Y0, phi, bctype)
```
* [cal_HansonKraus1991](./IHSetHansonKraus1991/calibration.py):
```python
# class that prepare the simulation framework
cal_HansonKraus1991(path)
```



## :package: Package structures
````

IHSetHansonKraus1991
|
├── LICENSE
├── README.md
├── build
├── dist
├── IHSetHansonKraus1991
│   ├── calibration.py
│   └── HansonKraus1991.py
└── .gitignore

````

---

## :incoming_envelope: Contact us
:snake: For code-development issues contact :man_technologist: [Lucas de Freitas](https://github.com/defreitasL) @ :office: [IHCantabria](https://github.com/IHCantabria)

## :copyright: Credits
Developed by :man_technologist: [Lucas de Freitas](https://github.com/defreitasL) @ :office: [IHCantabria](https://github.com/IHCantabria).
