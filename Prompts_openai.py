# Example prompt:
"""
I have some texts along with their corresponding evaluation indexs. The texts are arranged in ascending order
based on their weighted F1 score, which emphasize moer on recall than precision. I need you to look at recall, precision and accuracy in an integrated way, here higher scores indicate better quality.
text:
Let's figure it out!
score:
61
text:
Let's solve the problem.
score:
63
(. . . more instructions and scores . . . )
The following exemplars show how to apply your text: you replace <INS> in each input with your
text, then read the input and give an output. We say your output is wrong if your output is different
from the given output, and we say your output is correct if they are the same.
input:
Q: Alannah, Beatrix, and Queen are preparing for the new school year and have been given books
by their parents. Alannah has 20 more books than Beatrix. Queen has 1/5 times more books than
Alannah. If Beatrix has 30 books, how many books do the three have together?
A: <INS>
output:
140
(. . . more exemplars . . . )
Write your new text that is different from the old ones and has a score as high as possible. Write the
text in square brackets.
"""

meta_prompt = """
I have some prompt templates along with their corresponding evaluation indexs. The prompt templates are arranged in ascending order based on their F1 score. I need you to look at recall, precision, accuracy and F1 score in an integrated way, here higher scores indicate better quality.
prompt templates and their scores:
{texts_and_scores}
The following exemplars show how to apply your prompt template: you replace <prompt> with your prompt template, then read the question and give an output. We say your output is wrong if your output is different from the given output, and we say your output is correct if they are the same.
{exemplars}
Write your new prompt template that is different from the old ones and has a better performance in diagnosing depression which you think will get a f1_score as high as possible on the subsequent test set questions.
Because your answer will be directly used by me as a new template on the subsequent test set questions, so remember that you only need to generate a new prompt template, no other analysis is required.
Write the prompt template below in square brackets.
"""

scorer_prompt = """
Q: {prompt}
{question}
Now, I need you to analyze based on the the above user tweets, and finally give me your predicted label of the user status.
I want you to give me your answer in the following form:
""
### <Analysis>: "write your analysis of the user here",
### <Label>: "write your predicted label of the user here"
""
I want you to give me the predicted label in the form of a integer between 1 or 0(1 for "depression" and 0 for "control"). 
So in your answer, you should only give me an integer between 1 or 0 after the <Label> tag.
Even if you think it can't be classified, give directly the output integer between 1 or 0 which you think is closer.
"""

system_prompt = """
You are an expert in diagnosing depression. You have extensive experience in diagnosing depression and fantastic ability to construct prompt template. 
Next, I hope you can help me construct a better-performing prompt template according to the requirements. The specific requirements are as follows.
"""

# prompts = ["""Here are the tweets of a Twitter user and his tweeting time. This tweet data list contains a user's tweets over a long period of time, and the number of tweets is very large.
# I want you to try to classify the user's mental state between 1("depression") or 0("control") based on his list of tweets. 
# Below are the user's tweets and his tweeting time:""", 

# """Here are the tweets of a Twitter user and his tweeting time. This tweet data list contains a user's tweets over a long period of time, and the number of tweets is very large.
# I want you to think step by step. Try to classify the user's mental state between 1("depression") or 0("control") based on his list of tweets. 
# Below are the user's tweets and his tweeting time:""",

# """Here are the tweets of a Twitter user and his tweeting time. This tweet data list contains a user's tweets over a long period of time, and the number of tweets is very large.
# I want you to think step by step. Try to classify the user's mental state between 1("depression") or 0("control") based on his list of tweets. 
# Here are some key words of depression that may help you identify the user's mental state:
# 'depressed, sad, unhappy, miserable, sorrowful, dejected, downcast, downhearted, despondent, disconsolate, wretched, glum, gloomy, dismal, melancholy, woebegone, forlorn, crestfallen, heartbroken, inconsolable, grief-stricken, broken-hearted, heavy-hearted, heavy, low-spirited, morose, despairing, disheartened, discouraged, demoralized, desolate, despairing, desperate, anguished, distressed, fraught, distraught, suffering, wretched, traumatized, tormented, tortured, racked, haunted, hunted, persecuted, haunted, cursed, luckless, ill-fated, ill-starred, jinxed, unhappy, sad, miserable, sorrowful, dejected, depressed, down, downhearted, downcast, despondent, disconsolate, desolate, wretched, glum, gloomy, dismal, melancholy, woebegone, forlorn, crestfallen, broken-hearted, heartbroken, inconsolable, dispirited, discouraged, demoralized, down in the mouth, disheartened, despondent, dispirited, down, downcast, downhearted, down in the mouth, depressed, dejected, disconsolate, dispirited, discouraged, demoralized, desolate, disheartened, dispirited, discouraged, demoralized, despondent, disconsolate, downcast, downhearted, low-spirited, in the doldrums, despondent, disconsolate, down, downhearted, downcast, down in the mouth, depressed, dejected, dispirited, discouraged, demoralized, desolate, disheartened, dispirited, discouraged, demoralized, despondent, disconsolate, downcast, downhearted, low-spirited, in the doldrums, despondent, disconsolate, down, downhearted, downcast, down in the mouth, depressed, dejected, dispirited, discouraged, demoralized, desolate, disheartened, dispirited, discouraged, demoralized, despondent, disconsolate, downcast, downhearted, low-spirited, in the doldrums, despondent, discons'
# Below are the user's tweets and his tweeting time:"""
# ]

prompts = [
"""You are an expert in identifying depression. Next you need to help me diagnose depression based on user tweets. I want you to think step by step. 

Here are some knowledge frameworks about depression. Please read and use them in your subsequent judgments：

“**Individual Cognition and Thinking**:
Depression can involve cognitive challenges, including:

- Negative thought patterns
- Excessive worries
- Thoughts of self-harm
- Difficulties in concentrating or making decisions
Additionally, there might be feelings of despair or hopelessness about the future. The individual's positivity and optimism about life and the future are usually diminished.

**Emotional State**:
Key symptoms of depression encompass:

- Prolonged negative emotions
- Feelings of despair, guilt, worthlessness
- Loss of interest or pleasure in activities
Frequently, anxiety, fear, and concerns also accompany depression.

**Behavior and Willpower**:
Depression may lead to changes in an individual's behavior, such as:

- Social withdrawal
- Loss of interest in activities
- Alterations in appetite or sleep patterns
- Persistent fatigue
Often, individuals with depression might attempt to conceal or mask their true emotions or display emotions inconsistent with their internal feelings. This could extend to their language and expressions, where they might use a mix of positive and negative emotions, or ambiguous or vague language, to convey their sentiments and struggles. This may include self-care, seeking support, dealing with shame or discrimination, and more.

**Suicidal and Self-Harm Behavior**:
Consideration should be given to the presence of suicidal thoughts, self-harming behaviors, or other risky actions. These are severe manifestations of depression that require special attention.

**Life Events and Stressors**:
Understanding a patient's recent life events and experiences helps in determining if there are external factors that might trigger or exacerbate depression.”

Now after reading, below I will show you some tweets of a user. Please note that these tweets belong to the same user. Please analyze all tweets as a whole to determine the user's mental state rather than analyzing each tweet individually. 
"""
]
