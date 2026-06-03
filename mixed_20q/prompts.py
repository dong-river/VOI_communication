"""
Prompts and candidate sets for the 20-Questions and Medical-diagnosis games.

This is a trimmed, self-contained version that keeps only what the
Value-of-Information (yes/no) pipeline needs. Two domains are supported:

  * ``20q``     -- guess a hidden animal from a 100+ animal candidate set.
  * ``medical`` -- diagnose a hidden disease from a 52-disease candidate set.
"""

# --------------------------------------------------------------------------- #
# Batch answer-simulation prompts (used by the VoI belief / look-ahead model)
# --------------------------------------------------------------------------- #
ANIMAL_BATCH_ANSWER_SIM_PROMPT = """
You are playing a game of Twenty Questions. You will receive a yes/no question and a list of animals.
Your task is to answer the question for each animal with one of the following:
Yes - if the answer is clearly and scientifically correct.
No - if the answer is clearly and scientifically incorrect.
Maybe - if the answer is uncertain, ambiguous, varies by species/subspecies, is debated among experts, or if you are not confident. 

When in doubt, prefer "Maybe" over guessing Yes or No.

Format your response exactly as:
Animal Name: Yes/No/Maybe

Question:
{question}

Animals:
{candidate_list}
"""

MEDICAL_BATCH_ANSWER_SIM_PROMPT = """
I'm a medical diagnostician. Below is a yes/no question and a list of medical conditions.

Question:
"{question}"

For each condition, answer with just "Yes" or "No", based on typical symptoms and presentation.
Reply exactly as:
Condition: Answer

Conditions:
{candidate_list}
"""

ANIMAL_QUESTION_GENERATION_PROMPT = """
I'm playing a game of 20 Questions to identify an animal. Based on previous questions and answers:

{previous_qa}

Generate 5 yes/no questions that helps you identify the animal.
Format your response as a numbered list of questions.
"""

MEDICAL_QUESTION_GENERATION_PROMPT = """
I'm a doctor trying to diagnose a patient's condition through a series of questions. Based on the patient's symptoms and previous answers:

{previous_qa}

Generate 5 yes/no diagnostic questions that would most effectively narrow down the possible conditions.
Format your response as a numbered list of questions.
"""

INFO_GAIN_PROMPTS = {
    '20q': {
        'question_generation': ANIMAL_QUESTION_GENERATION_PROMPT,
        'batch_answer_simulation': ANIMAL_BATCH_ANSWER_SIM_PROMPT,
    },
    'medical': {
        'question_generation': MEDICAL_QUESTION_GENERATION_PROMPT,
        'batch_answer_simulation': MEDICAL_BATCH_ANSWER_SIM_PROMPT,
    },
}

# --------------------------------------------------------------------------- #
# Candidate sets
# --------------------------------------------------------------------------- #
LARGE_ANIMAL_ANSWER_SET = [
    'elephant', 'giraffe', 'zebra', 'gorilla', 'chimpanzee', 'panda', 'koala', 'kangaroo',
    'flamingo', 'peacock', 'crocodile', 'alligator', 'tortoise', 'chameleon', 'salamander', 'newt',
    'shark', 'dolphin', 'whale', 'octopus', 'squid', 'lobster', 'crab', 'crab', 'shark', 'dolphin', 'whale', 'octopus', 'squid', 'lobster', 'crab',
    'bear', 'wolf', 'skunk', 'badger', 'otter', 'beaver', 'chipmunk', 'moose', 'elk',
    'bison', 'buffalo', 'rhinoceros', 'hippopotamus', 'camel', 'llama', 'alpaca', 'hedgehog', 'armadillo',
    'jaguar', 'leopard', 'panther', 'cheetah', 'cougar', 'lynx', 'bobcat', 'hyena', 'jackal', 'dingo',
    'meerkat', 'mongoose', 'weasel', 'mink', 'ferret', 'marten', 'wolverine', 'polecat', 'stoat',
    'ocelot', 'serval', 'caracal', 'margay', 'jaguarundi', 'puma', 'leopard seal', 'walrus', 'sea lion',
    'manatee', 'dugong', 'narwhal', 'beluga', 'orca', 'porpoise', 'humpback whale', 'blue whale', 'sperm whale', 'minke whale',
    'right whale', 'bowhead whale', 'fin whale', 'sei whale', "bryde's whale", 'gray whale', 'pilot whale', 'beaked whale', 'pygmy whale', 'baleen whale',
    'marlin', 'swordfish', 'sailfish', 'barracuda', 'halibut', 'clownfish', 'angelfish', 'piranha',
    'stingray', 'manta ray', 'seahorse', 'pufferfish', 'arapaima', 'angler fish', 'moray eel',
    'arowana', 'leatherback', 'loggerhead', 'hawksbill',
    'python', 'anaconda', 'boa constrictor', 'viper', 'cobra', 'rattlesnake', 'mamba', 'copperhead', 'adder', 'coral snake',
    'cottonmouth', 'kingsnake', 'taipan', 'krait', 'gila monster', 'monitor lizard', 'komodo dragon', 'skink',
    'tuatara', 'horned lizard', 'frilled lizard', 'alligator lizard', 'worm lizard', 'glass lizard', 'agama', 'tegu',
    'uromastyx', 'axolotl', 'caecilian', 'olm', 'siren', 'mudpuppy', 'hellbender', 'poison dart frog',
    'spadefoot toad', 'cane toad', 'surinam toad', 'fire-bellied toad', 'clawed frog', 'glass frog', 'goliath frog', 'flying frog',
    'albatross', 'pelican', 'heron', 'stork', 'crane', 'ibis', 'spoonbill', 'cormorant', 'egret',
    'falcon', 'kestrel', 'condor', 'vulture', 'osprey', 'kite', 'harrier', 'secretary bird',
    'jackdaw', 'toucan', 'hornbill', 'hummingbird', 'kingfisher', 'nightingale', 'lark', 'swift', 'nightjar',
    'ptarmigan', 'ostrich', 'emu', 'cassowary', 'kiwi', 'rhea', 'cockatoo', 'macaw',
    'lorikeet', 'kookaburra', 'bittern', 'loon', 'grebe', 'rail',
    'bustard', 'sandpiper', 'curlew', 'avocet', 'oystercatcher', 'lapwing', 'tern',
    'skimmer', 'auk', 'puffin', 'guillemot', 'murre', 'quetzal', 'roadrunner', 'trogon', 'hoopoe',
    'okapi', 'tapir', 'aardvark', 'pangolin', 'anteater', 'sloth',
    'gazelle', 'impala', 'springbok', 'wildebeest', 'kudu', 'nyala', 'oryx', 'eland', 'dik-dik',
    'gerenuk', 'ibex', 'markhor', 'chamois', 'tahr', 'bongo', 'waterbuck', 'reedbuck', 'kob', 'roan antelope',
    'sable antelope', 'gemsbok', 'hartebeest', 'bontebok', 'blesbuck', 'bushbuck', 'sitatunga', 'klipspringer', 'duiker', 'steenbok',
    'lemur', 'loris', 'galago', 'tarsier', 'marmoset', 'tamarin', 'capuchin', 'howler monkey', 'spider monkey',
    'mandrill', 'colobus', 'langur', 'proboscis monkey', 'gibbon', 'orangutan', 'bonobo',
    'lemming', 'vole', 'shrew', 'mole-rat', 'chinchilla', 'degu', 'capybara', 'paca',
    'agouti', 'coypu', 'hutia', 'mara', 'viscacha', 'jerboa', 'springhare', 'zokor', 'gundi',
    'hyrax', 'cuscus', 'quokka', 'numbat', 'bilby', 'bandicoot',
    'quoll', 'thylacine', 'tasmanian devil', 'wombat', 'sugar glider', 'colugo', 'aye-aye', 'tenrec',
    'babirusa', 'warthog', 'peccary', 'okapi', 'yak', 'gaur', 'banteng',
    'wisent', 'aurochs', 'zebu', 'takin', 'muskox', 'addax', 'saiga', 'argali', 'urial',
    'mouflon', 'aoudad', 'nilgai', 'blackbuck', 'chiru', 'saola', 'serow', 'goral', 'tamaraw',
    'pronghorn', 'vicuna', 'guanaco', 'red panda', 'musk deer', 'pudu', 'muntjac', 'sambar', 'chital',
    'huemul', 'tule elk', 'wapiti', 'sika deer', 'barasingha',
]

def _dedup(lst):
    seen = set(); out = []
    for x in lst:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

LARGE_ANIMAL_ANSWER_SET_DEDUPED = _dedup(LARGE_ANIMAL_ANSWER_SET)

ANIMAL_ANSWER_SET_LIST = {
    'large': LARGE_ANIMAL_ANSWER_SET_DEDUPED,
    'animal_set': LARGE_ANIMAL_ANSWER_SET_DEDUPED[:100],
    'default': ['dog', 'cat', 'horse', 'elephant', 'lion', 'tiger', 'giraffe', 'zebra', 'monkey', 'gorilla',
                'chimpanzee', 'panda', 'koala', 'kangaroo', 'penguin', 'eagle', 'hawk', 'owl', 'parrot', 'flamingo',
                'duck', 'goose', 'swan', 'chicken', 'rooster', 'turkey', 'peacock', 'crocodile', 'alligator', 'turtle',
                'tortoise', 'snake', 'lizard', 'iguana', 'chameleon', 'frog', 'toad', 'salamander', 'newt', 'fish',
                'shark', 'dolphin', 'whale', 'octopus', 'squid', 'lobster', 'crab'],
}

ANIMAL_ANSWER_SET = ANIMAL_ANSWER_SET_LIST['large']

MEDICAL_ANSWER_SET = [
    'Enteritis', 'Gastritis', 'Gastroenteritis', 'Esophagitis', 'Cholecystitis',
    'Appendicitis', 'Pancreatitis', 'Gastric ulcer', 'Constipation', 'Cold',
    'Irritable bowel syndrome', 'Diarrhea', 'Allergic rhinitis', 'Upper respiratory tract infection',
    'Pneumonia', 'Hepatitis', "Crohn's disease", 'Ulcerative colitis', 'Lactose intolerance',
    'Bronchitis', 'Asthma', 'Sinusitis', 'Chronic obstructive pulmonary disease (COPD)', 'Influenza',
    'COVID-19', 'Strep throat', 'Mononucleosis', 'Tuberculosis', 'Urinary tract infection',
    'Pyelonephritis', 'Kidney stones', 'Migraine', 'Tension headache', 'Meningitis',
    'Epilepsy', 'Diabetes', 'Hyperthyroidism', 'Hypothyroidism', 'Osteoarthritis',
    'Rheumatoid arthritis', 'Fibromyalgia', 'Eczema', 'Psoriasis', 'Contact dermatitis',
    'Shingles', 'Hypertension', 'Coronary artery disease', 'Myocardial infarction',
    'Heart failure', 'Depression', 'Anxiety disorder', 'Bipolar disorder',
]

# --------------------------------------------------------------------------- #
# Animal (20q) prompts
# --------------------------------------------------------------------------- #
ANIMAL_FORCE_GUESS_PROMPT = """You are tasked with guessing the animal based on the entire conversation. You must now provide a final guess, even if you are uncertain.
The set of possible animals is: {answer_set}.

You must also estimate your confidence in the guess on a scale from 1 to 100, where 100 indicates the highest confidence.
Directly output a JSON object in the following format without any reasoning before or after:  
{{"guess": "<animal>", "confidence": <number between 1 and 100>}}

Confidence Scale (1-100):
1-20: Very minimal evidence - the guess is essentially a shot in the dark, with no clear signals or clues.
21-40: Weak evidence - there are a few weak signals, but the information is vague or matches many different animals.
41-60: Moderate evidence - there are some partial matches or indirect clues, but the guess still involves significant uncertainty.
61-80: Strong evidence - there are several consistent clues that point toward a small set of likely animals.
81-100: Very strong evidence - there are clear, specific clues that strongly indicate a particular animal, resulting in high confidence.
"""

ANIMAL_FINAL_GUESS_REQUEST = """Here's the conversation history:
{conversation_history}

It's time to make your final guess.
"""

ANIMAL_ANSWERER_PROMPT = """You are playing 20 Questions as the answerer. You are thinking of this animal: {subject}.

Rules:
1. Answer the questions truthfully about {subject}.
2. For yes/no questions, you must respond with either "Yes", "No", or "Maybe" (if you're uncertain).
3. Most importantly, do not directly reveal the animal unless they guess correctly.
"""

ANIMAL_ANSWERER_PROMPT_FREE = """You are playing 20 Questions as the answerer. You are thinking of this animal: {subject}.

Rules:
1. Answer the questions truthfully about {subject}.
2. Provide detailed but concise answers to help the guesser.
3. Do not directly reveal the animal unless they guess correctly.
4. Be factual and accurate in your responses.
"""

ANIMAL_GUESSER_PROMPT_YES_NO = """You are playing 20 Questions as the guesser. Your goal is to figure out what animal I'm thinking of by asking questions.

You have asked {question_count} questions so far and have {remaining_questions} questions left.

The possible animals you're trying to guess include: {answer_set}

Rules:
1. Ask only yes/no questions (questions that can be answered with "Yes" or "No").
2. Ask one question at a time.
3. Keep asking questions until you use up all your 20 questions.
4. Do not ask the same question twice.
5. Do not guess the specific animal (e.g., "Is it a cat?"). Ask broader category-level questions to narrow down the options
"""

# --------------------------------------------------------------------------- #
# Medical prompts
# --------------------------------------------------------------------------- #
MEDICAL_FORCE_GUESS_PROMPT = """You are tasked with diagnosing the patient based on the entire conversation. You must now provide a final diagnosis, even if you are uncertain.
The set of possible diseases is: {answer_set}.

You must also estimate your confidence in the diagnosis on a scale from 1 to 100, where 100 indicates the highest confidence.  
Directly output a JSON object in the following format without any reasoning as preambles:  
{{"guess": "<diagnosis>", "confidence": <number between 1 and 100>}}

Confidence Scale (1-100):
1-30: Extremely uncertain - virtually no useful medical evidence.
31-50: Weak evidence - only a few nonspecific symptoms that could match many diseases.
51-70: Moderate evidence - several relevant symptoms are present, but the diagnosis could still fit multiple diseases.
71-90: Strong evidence - multiple consistent symptoms strongly favor one or two likely diseases.
91-100: Very strong evidence - the symptoms are clear, specific, and strongly indicate a particular disease."""

MEDICAL_FINAL_GUESS_REQUEST = """Here's the conversation history:
{conversation_history}

It's time to make your final diagnosis.
"""

MEDICAL_ANSWERER_PROMPT = """You are the patient suffering '{subject}' and I am the doctor.
Here is your current self-report:
{self_repo}

Limit your answer to "Yes" or "No". Do not mention the disease name directly unless I correctly guess it.
"""

MEDICAL_ANSWERER_PROMPT_FREE = """You are the patient suffering '{subject}' and I am the doctor.
Here is your conversation history with another doctor about the same condition:
{conv_hist}

Please use this conversation as context to answer my questions truthfully and consistently based on your disease.
Answer in a similar style and manner as shown in the conversation history.

Do not mention the disease name directly unless I correctly guess it.
"""

MEDICAL_GUESSER_PROMPT_YES_NO = """You are the doctor and you must diagnose the patient using only yes/no questions.

You have asked {question_count} questions so far and have {remaining_questions} questions left.

The possible diagnoses you're considering include: {answer_set}

You can ask up to 20 yes/no questions to understand the patient's condition.

At the end, you must guess what disease the patient is suffering from.

# Example:
Doctor: Have you had a fever recently?
Patient: Yes.
Doctor: Are you experiencing a sore throat?
Patient: Yes.
Doctor: Do you have a cough?
Patient: No.
"""

# --------------------------------------------------------------------------- #
# Domain prompt registry
# --------------------------------------------------------------------------- #
ANIMAL_PROMPTS = {
    "answer_set": ANIMAL_ANSWER_SET,
    "force_guess": ANIMAL_FORCE_GUESS_PROMPT,
    "final_guess_request": ANIMAL_FINAL_GUESS_REQUEST,
    "answerer": ANIMAL_ANSWERER_PROMPT,
    "answerer_free": ANIMAL_ANSWERER_PROMPT_FREE,
    "guesser": {
        "yes_no": ANIMAL_GUESSER_PROMPT_YES_NO,
        "free": ANIMAL_ANSWERER_PROMPT_FREE,
    },
}

MEDICAL_PROMPTS = {
    "answer_set": MEDICAL_ANSWER_SET,
    "force_guess": MEDICAL_FORCE_GUESS_PROMPT,
    "final_guess_request": MEDICAL_FINAL_GUESS_REQUEST,
    "answerer": MEDICAL_ANSWERER_PROMPT,
    "answerer_free": MEDICAL_ANSWERER_PROMPT_FREE,
    "guesser": {
        "yes_no": MEDICAL_GUESSER_PROMPT_YES_NO,
        "free": MEDICAL_GUESSER_PROMPT_YES_NO,
    },
}

PROMPTS = {
    "20q": ANIMAL_PROMPTS,
    "medical": MEDICAL_PROMPTS,
}

# Per-disease reward used to weight the medical task (higher = more severe).
reward_dict = {
    'Enteritis': 5, 'Gastritis': 5, 'Gastroenteritis': 6, 'Esophagitis': 5, 'Cholecystitis': 7,
    'Appendicitis': 8, 'Pancreatitis': 9, 'Gastric ulcer': 7, 'Constipation': 4, 'Cold': 3,
    'Irritable bowel syndrome': 5, 'Diarrhea': 4, 'Allergic rhinitis': 2,
    'Upper respiratory tract infection': 5, 'Pneumonia': 8, 'Hepatitis': 8,
    "Crohn's disease": 8, 'Ulcerative colitis': 8, 'Lactose intolerance': 4, 'Bronchitis': 5,
    'Asthma': 7, 'Sinusitis': 4, 'Chronic obstructive pulmonary disease (COPD)': 8, 'COVID-19': 8,
    'Strep throat': 6, 'Mononucleosis': 6, 'Tuberculosis': 9, 'Urinary tract infection': 6,
    'Pyelonephritis': 8, 'Kidney stones': 7, 'Migraine': 5, 'Tension headache': 4, 'Meningitis': 9,
    'Epilepsy': 8, 'Diabetes': 8, 'Hyperthyroidism': 7, 'Hypothyroidism': 7, 'Osteoarthritis': 5,
    'Rheumatoid arthritis': 7, 'Fibromyalgia': 5, 'Eczema': 4, 'Psoriasis': 4, 'Contact dermatitis': 1,
    'Shingles': 6, 'Hypertension': 8, 'Coronary artery disease': 9, 'Myocardial infarction': 10,
    'Heart failure': 9, 'Depression': 7, 'Anxiety disorder': 6, 'Bipolar disorder': 7,
}
