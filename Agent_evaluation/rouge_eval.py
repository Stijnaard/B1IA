from rouge import Rouge 

with open('netflix_testsum.txt', 'r', encoding='utf-8') as file:
    AI_summary = file.read()

with open('Netflix_Annotator1_Noortje.txt', 'r', encoding='utf-8') as file:
    Annotator_summary = file.read() 

# hypothesis = "the #### transcript is a written version of each day 's cnn student news program use this transcript to he    lp students with reading comprehension and vocabulary use the weekly newsquiz to test your knowledge of storie s you     saw on cnn student news"

# reference = "this page includes the show transcript use the transcript to help students with reading comprehension and     vocabulary at the bottom of the page , comment for a chance to be mentioned on cnn student news . you must be a teac    her or a student age # # or older to request a mention on the cnn student news roll call . the weekly newsquiz tests     students ' knowledge of even ts in the news"

rouge = Rouge()
scores = rouge.get_scores(AI_summary, Annotator_summary)
print(scores)