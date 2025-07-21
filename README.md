# 🌐 Simulating Society: Leveraging Large Language Models as Citizen Agents to Study Urban Behavior

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Made at MIT Media Lab](https://img.shields.io/badge/MIT-Media--Lab-red.svg)](https://www.media.mit.edu/groups/city-science/overview/)

> **Master’s thesis project at the [MIT Media Lab – City Science Group](https://www.media.mit.edu/groups/city-science/overview/)**


---

## 🧠 Overview
This project explores the generation and fine-tuning of **synthetic human profiles** using Large Language Models (LLMs), enriched with demographic realism and **psychological depth** via the **Big Five personality traits**. It was developed as part of a **Master’s thesis** at the [MIT Media Lab – City Science Group](https://www.media.mit.edu/groups/city-science/overview/).

We introduce a **two-phase methodology**:

### 🔹 Phase I: Synthetic Profile Generation
- Generate rich human profiles from real demographic data (Cambridge, MA) + Big Five traits.
- Evaluate four LLMs: **LLaMA**, **Qwen**, **Mistral**, and **Dolphin**.
- Output: Structured JSON profiles and scripts for semantic, lexical, and statistical analysis.

### 🔹 Phase II: Personality-Based Fine-Tuning
- Build a custom dataset from 200+ topic-aligned questions and LLM answers.
- Fine-tune LLaMA 3 8B using **LoRA** + **Direct Preference Optimization (DPO)** to match personality vectors.
- Evaluate control over Openness, Extraversion, etc., and assess cultural/emotional biases.

## 📁 Repository Structure


- **Phase_I/**  
  This folder contains the scripts and data related to the first phase of the project:
  - `data/`: Stores raw demographic and personality data files such as Big Five parameters and Cambridge demographic JSONs.
  - `generator.py`: Main script to generate synthetic profiles using LLMs.
  - `profile_utils.py`: Utility functions supporting profile generation and handling.
  - `profiles/`: Contains JSON files with generated synthetic profiles from different LLMs and configurations.
  - `requirements.txt`: Dependencies for Phase I.

- **Phase_II/**  
  This folder contains the dataset creation, fine-tuning, and evaluation pipeline:
  - `1_Generate_Questions_by_200_Topics/`: Scripts and datasets for generating personality-related questions categorized by topics.
  - `2_Answering_Questions/`: Scripts for generating responses based on the questions.
  - `3_Create_Datasets_By_Answers/`: Converts answers into structured datasets for training and includes dataset analysis and visualization.
  - `4_Models_Train/`: Scripts to train and evaluate fine-tuned models.
  - `5_Models_Test/`: Testing scripts and generated responses for analyzing fine-tuned model outputs.
  - `requirements.txt`: Dependencies for Phase II.

- **Results/**  
  Stores evaluation scripts, plots, and results of the thesis:
  - `Phase I/`: Contains scripts and plots related to semantic, sentiment, variable analysis, and word clouds of generated profiles.
  - `Phase II/`: Contains plots and scripts for evaluating fine-tuned model personality alignments and sentiment analysis.

## 🚀 How to Use

### 🧬 Phase I – Generate Profiles
```bash
cd Phase_I
pip install -r requirements.txt
python python generator.py <LLM_MODEL>
```
Edit `generator.py` or supply config to customize demographics, number of profiles, or target traits.

### 🔧 Phase II – Dataset & Fine-Tuning
```bash
cd Phase_II
pip install -r requirements.txt
# Follow folders 1_ to 5_ in order:
# - Generate questions
# - Collect LLM answers
# - Create train-ready datasets
# - Train and evaluate fine-tuned models
# - Run tests on output
```



## 📦 Dependencies
Each phase contains a `requirements.txt` with specific Python packages.
We recommend using isolated environments (e.g., `venv` or `conda`) for each phase.

## 🎯 Use Cases
- Human-centered agent-based simulations (urban planning, mobility, policy-making)
- Synthetic data for behavioral science, psychology, education
- Testing bias, alignment, and safety in LLMs
- Emotionally nuanced conversational agents

## 📚 Citation & Contact
This project was developed as part of a Master's Thesis at the **MIT Media Lab**.

🔗 GitHub: [CityScope/personality-driven-synthetic-populations](https://github.com/CityScope/personality-driven-synthetic-populations)

📄 For academic use, please cite the accompanying thesis. [Download Thesis (PDF)](./doc/JoseMiguel_Nicolas_thesis_MIT_MediaLab_2025.pdf)

> **📍 Note:** This thesis was submitted to the *Universidad Politécnica de Madrid (UPM)* as part of a Master's degree.  
> The full research was conducted at the *MIT Media Lab – City Science Group* in collaboration with their research team.
---

> *"Computing is not about computers anymore. It is about living."*  
> — *Nicholas Negroponte, MIT Media Lab Founder*
