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
I have some texts along with their corresponding evaluation indexs. The texts are arranged in ascending order
based on their weighted F1 score, which emphasizes moer on recall than precision. 
I need you to look at recall, precision and accuracy in an integrated way, here higher scores indicate better quality.
text:
{texts_and_scores}
The following exemplars show how to apply your text: you replace <INS> in each input with your
text, then read the input and give an output. We say your output is wrong if your output is different
from the given output, and we say your output is correct if they are the same.
{exemplars}
Write your new text that is different from the old ones and has a score as high as possible. Write the
text in square brackets.
"""

scorer_prompt = """
Q: {instruction}
{question}
Since your output will be used directly as predicted labels, so just give me the precise output in the form of a integer between 1 or 0(1 for "depression" and 0 for "control").
Even if you believe the user can't be classified, just give directly the answer integer between 1 or 0 which you think is closer.
A:
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
    """You are an expert in identifying depression. I will give you the tweets of a user. I want you to help me diagnose whether the user is depression based on the given text.
The user's tweets will be given after "text:". Fill in the <ans> part use 1 or 0, where 1 represents for depression and 0 represents for coontrol. Just don't give me the output of None.
Here are some examples that you should try to give me your output exactly in this format.
text: "[1373456856, "I prefer couldn't've"], [1373463920, 'This needs to be added to the [FFF system]],"
output: 1
text: "[1396277992, 'Stupid exempt status. Our new policy is that we "donate" 5 hrs a week before we get paid overtime. At least we get paid overtime after that, I guess'], [1396280820, "not if you're an exempt worker! woo-hoo!"],"
output: 0
Because the final output you gave me will be used directly by me as the prediction label, so do not output your analysis content, just output 1 or 0.
Even if you think you don't have enough information to classify, just give directly the output integer between 1 or 0 which you think is closer:
text: {question}
output: <ans>
""",

"""You are an expert in identifying depression. Next you need to help me diagnose depression based on user tweets.
Here are some examples that you should try to give me your output exactly in this format.
text: "tweets of the user"
Output: 1
text: "tweets of the user"
Output: 0
Now there are some tweets of a user. Fill in the <ans> part use 1 or 0, where 1 represents for depression and 0 represents for coontrol.
Because the final output you gave me will be used directly by me as the prediction label, so do not output your analysis content, just output 1 or 0.
Even if you think you need more information to accurately identify depression in these tweets, give directly the output integer between 1 or 0 which you think is closer:
text: {question}
output: <ans>
""",

"""You are an expert in identifying depression. Next you need to help me diagnose depression based on user tweets. I want you to think step by step.
Here are some examples that you should try to give me your output exactly in this format.
text: "[1373456856, "I prefer couldn't've"], [1373463920, 'This needs to be added to the [FFF system](http://en.wikipedia.org/wiki/FFF_system)'],"
Output: 1
text: "[1396277992, 'Stupid exempt status. Our new policy is that we "donate" 5 hrs a week before we get paid overtime. At least we get paid overtime after that, I guess'], [1396280820, "not if you're an exempt worker! woo-hoo!"],"
output: 0
Now there are some tweets of a user, I want you to think step by step. Fill in the <ans> part use 1 or 0, where 1 represents for depression and 0 represents for coontrol. Just don't give me the output of None.
Because the final output you gave me will be used directly by me as the prediction label, so do not output your analysis content, just output 1 or 0.
Even if you think it can't be classified, give directly the output integer between 1 or 0 which you think is closer:
text: {question}
output: <ans>
""",

"""You are an expert in identifying depression. Next you need to help me diagnose depression based on user tweets. I want you to think step by step.
Here are some examples that you should try to give me your output exactly in this format.
text: "[1373456856, "I prefer couldn't've"], [1373463920, 'This needs to be added to the [FFF system](http://en.wikipedia.org/wiki/FFF_system)'],"
Output: 1
text: "[1396277992, 'Stupid exempt status. Our new policy is that we "donate" 5 hrs a week before we get paid overtime. At least we get paid overtime after that, I guess'], [1396280820, "not if you're an exempt worker! woo-hoo!"],"
output: 0
Here are some key words of depression that may help you identify the user's mental state:
'depressed, sad, unhappy, miserable, sorrowful, dejected, downcast, downhearted, despondent, disconsolate, wretched, glum, gloomy, dismal, melancholy, woebegone, forlorn, crestfallen, heartbroken, inconsolable, grief-stricken, broken-hearted, heavy-hearted, heavy, low-spirited, morose, despairing, disheartened, discouraged, demoralized, desolate, despairing, desperate, anguished, distressed, fraught, distraught, suffering, wretched, traumatized, tormented, tortured, racked, haunted, hunted, persecuted, haunted, cursed, luckless, ill-fated, ill-starred, jinxed, unhappy, sad, miserable, sorrowful, dejected, depressed, down, downhearted, downcast, despondent, disconsolate, desolate, wretched, glum, gloomy, dismal, melancholy, woebegone, forlorn, crestfallen, broken-hearted, heartbroken, inconsolable, dispirited, discouraged, demoralized, down in the mouth, disheartened, despondent, dispirited, down, downcast, downhearted, down in the mouth, depressed, dejected, disconsolate, dispirited, discouraged, demoralized, desolate, disheartened, dispirited, discouraged, demoralized, despondent, disconsolate, downcast, downhearted, low-spirited, in the doldrums, despondent, disconsolate, down, downhearted, downcast, down in the mouth, depressed, dejected, dispirited, discouraged, demoralized, desolate, disheartened, dispirited, discouraged, demoralized, despondent, disconsolate, downcast, downhearted, low-spirited, in the doldrums, despondent, disconsolate, down, downhearted, downcast, down in the mouth, depressed, dejected, dispirited, discouraged, demoralized, desolate, disheartened, dispirited, discouraged, demoralized, despondent, disconsolate, downcast, downhearted, low-spirited, in the doldrums, despondent, discons'
Now there are some tweets of a user, I want you to think step by step to help me analyse these tweets just like the above examples. Fill in the <ans> part use 1 or 0, where 1 represents for depression and 0 represents for coontrol. Just don't give me the output of None.
Because the final output you gave me will be used directly by me as the prediction label, so do not output your analysis content, just output 1 or 0.
Even if you think it can't be classified, give directly the output integer between 1 or 0 which you think is closer:
text: {question}
output: <ans>
""",



]
