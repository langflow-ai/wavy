<!-- Title -->
## ≋ Wavy: Time-Series Manipulation ≋

<p>
<img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/logspace-ai/wavy" />
<img alt="GitHub Last Commit" src="https://img.shields.io/github/last-commit/logspace-ai/wavy" />
<!-- <img alt="GitHub Language Count" src="https://img.shields.io/github/languages/count/logspace-ai/wavy" /> -->
<img alt="" src="https://img.shields.io/github/repo-size/logspace-ai/wavy" />
<!-- <img alt="GitHub Issues" src="https://img.shields.io/github/issues/logspace-ai/wavy" /> -->
<!-- <img alt="GitHub Closed Issues" src="https://img.shields.io/github/issues-closed/logspace-ai/wavy" /> -->
<!-- <img alt="GitHub Pull Requests" src="https://img.shields.io/github/issues-pr/logspace-ai/wavy" /> -->
<!-- <img alt="GitHub Closed Pull Requests" src="https://img.shields.io/github/issues-pr-closed/logspace-ai/wavy" />  -->
<!-- <img alt="GitHub Commit Activity (Year)" src="https://img.shields.io/github/commit-activity/y/logspace-ai/wavy" /> -->
<img alt="Github License" src="https://img.shields.io/github/license/logspace-ai/wavy" />  
</p>


Wavy is a time series package designed to simplify the pre-processing steps and reliably avoid the problem of data leakage on time-series data. <a href="https://logspace-ai.github.io/wavy/"><strong>Explore the docs 📖</strong></a>
    <a href="https://github.com/logspace-ai/wavy">
        <img width="50%" src="https://github.com/logspace-ai/wavy/blob/main/images/logo.png?raw=true" alt="Logo" width="419" height="235" align="right"></a>

  

<!-- PROJECT LOGO -->
<!-- <div align="right">
  <a href="https://github.com/logspace-ai/wavy">
    <img width="49%" src="https://github.com/logspace-ai/wavy/blob/main/images/logo.png?raw=true" alt="Logo" width="419" height="235">
  </a>

</div> -->

<!-- GETTING STARTED -->
## 📦 Installation

You can install Wavy from pip:

```bash
pip install wavyts
```


<!-- GETTING STARTED -->
## 🚀 Quickstart

```python
import numpy as np
import pandas as pd
import wavy

# Start with any time-series dataframe
df = pd.DataFrame({'price': np.random.randn(1000)}, index=range(1000))
df.head()
```

Out[ ]:	
price
0	-0.495187
1	-1.370577
2	0.637780
3	-0.922224
4	0.551576


<!-- Description -->
## Features

💡 Wavy **is**:

- A **resourceful**, **high-level** package with tools for time-series processing, visualization, and modeling.
- A facilitator for **time-series windowing** that helps reduce boilerplate code and avoid shape confusion.

❗ Wavy **is not**:

- An efficient, performance-first framework (**yet!**).
- Primarily focused on models. Processed data can be easily converted to numpy arrays for further exploration.


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make to `wavy` are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! ⭐

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/logspace-ai/wavy.svg?style=for-the-badge
[contributors-url]: https://github.com/logspace-ai/wavy/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/logspace-ai/wavy.svg?style=for-the-badge
[forks-url]: https://github.com/logspace-ai/wavy/network/members
[stars-shield]: https://img.shields.io/github/stars/logspace-ai/wavy.svg?style=for-the-badge
[stars-url]: https://github.com/logspace-ai/wavy/stargazers
[issues-shield]: https://img.shields.io/github/issues/logspace-ai/wavy.svg?style=for-the-badge
[issues-url]: https://github.com/logspace-ai/wavy/issues
[license-shield]: https://img.shields.io/github/license/logspace-ai/wavy.svg?style=for-the-badge
[license-url]: https://github.com/logspace-ai/wavy/blob/main/LICENSE.txt
<!-- [documentation-url]: https://logspace-ai.github.io/wavy/ -->
