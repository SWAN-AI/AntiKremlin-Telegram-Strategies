## AntiKremlin-Telegram-Strategies

This repository contains the code related to the research paper titled "The Adaptive Strategies of Anti-Kremlin Digital Dissent in Telegram during the Russian Invasion of Ukraine." The study examines online communications amid the Russia-Ukraine conflict, analyzing 114 anti-Kremlin channels with more than 1 million posts and the corresponding viewer reactions. 


### Dataset

Data used to produce the findings reported in this study can be accessible from: https://www.dropbox.com/scl/fo/lwhr16nbxej4kphrajjjq/AHdV-FJBoiT-csTZh4j4x1E?rlkey=q0vuosiod4469jqec2fp5whsl&dl=0

### Contents

model.py: File includes multilingual MPNet language model to generate embeddings of the Russian language and create topical clusters for each phase.

post_volume_analysis.ipynb: This file conducts a statistical analysis of post volumes within the top 5 prominent categories.

user_reactions_analysis.ipynb: This file performs a statistical analysis of emoji reactions to posts within the top 5 prominent categories.