# Knowledge-enhanced-Prompting
We proposed a depression knowledge-driven prompt engineering approach named "KePrompt" (Knowledge-enhanced Prompting). Our extensive dataset includes data from over 30,000 users, capturing their tweets over a period of up to ten years. Each user's contribution averages more than 7,000 words, providing a rich resource for depression analysis. 

## Method
![image](https://github.com/Connor-Shen/Knowledge-enhanced-Prompting/blob/main/img/experiment_structure.png)


## Highlights
• Proposing a knowledge-driven prompt engineering approach, which leverages prompt engineering to help analyze users’ tweets over a period of up to ten years, and assess whether the user's mental state indicates depression.

• Utilizing LLMs as both optimizer and scorer to optimize prompts, while also incorporating human experts as assist scorer. Through iterative rounds, this approach generates the most effective prompt for analyzing depression. 

• Employing optimized prompt templates to explore knowledge embedded within LLMs (intrinsic knowledge), and concurrently incorporating expert-provided domain knowledge (extrinsic knowledge) to construct a valuable domain knowledge framework.


## LLM knowledge-task pairs
### What do LLMs know?
<img src="https://github.com/Connor-Shen/Knowledge-enhanced-Prompting/blob/main/img/LLM_knowledge.png" width="100px">
### What we do?
![image](https://github.com/Connor-Shen/Knowledge-enhanced-Prompting/blob/main/img/Know_knows.png)
