"""
generator.py
Main script for generating synthetic human profiles and reflective tasks using an LLM.

This script orchestrates the two-phase pipeline:
1. Stochastic profile generation using `fill_taxonomy()`.
2. Reflection tasks and enrichment using a conversational LLM API.

Each profile is completed, enriched, and saved in both JSON and TXT formats.

Usage:
    $ python main.py <LLM_MODEL>

Example:
    $ python main.py llama3.1:8b

Author: José Miguel Nicolás García
"""


import requests
import json
import sys
from profile_utils import fill_taxonomy



# === Configuration Variables ===

NUMBER_OF_PROFILES = 1180 # Total number of synthetic profiles to generate in the simulation. A value of 1,180 corresponds to a 1:100 representation of ~118,000 people.
URL = "http://localhost:11436/api/chat" # URL of the local LLM API endpoint (Ollama is used in this case).

MODEL = sys.argv[1] if len(sys.argv) == 2 else None  # Read model name from command-line argument (e.g., "llama3.1:8b")
if MODEL is None:
    print("Tell me the LLM model please.")
    sys.exit(1)

# Output file paths for storing generated profiles and their enriched reflections
PROFILES_JSON_PATH = f'./profiles/profiles_cambridge-{MODEL}.json'
REFLECTIONS_JSON_PATH = f'./profiles/profiles_cambridge_reflexions-{MODEL}.json'


def main():

    """
    Complete pipeline. Generates and enriches synthetic profiles
    through a multi-step interaction with an LLM. Outputs are saved in JSON format.


    Steps per iteration:
        1. Generate base profile using fill_taxonomy().
        2. Construct system prompt and send it to the LLM.
        3. Ask 8 predefined reflection tasks.
        4. Request a fully filled JSON structure from the LLM.
        5. Save the final result in multiple formats.

    Raises:
        SystemExit: If the model name is not provided via command-line argument.
    """
    if len(sys.argv) != 2:
        print("Tell me the LLM model please.")
        sys.exit(1)

    # === LLM Reflection Tasks ===
    # These are sequential prompts sent to the LLM to simulate reasoning and profile enrichment

    tasks=[]
    #Task1
    tasks.append(
        """*Talk and reflexion about the personality scores**
    Generating a detailed personality profile based on the Big 5 Personality Scores. These scores describe the five key dimensions of human personality and range from **0** (minimum presence of a trait) to **1** (maximum presence of a trait). 

    Here is a description of the Big 5 Personality Traits and how they influence personality:

    Openness to Experience:
    - High (0.75-1.0): Curious, imaginative, artistic, open to new ideas.
    - Medium (0.35-0.74): Practical but somewhat open to novelty.
    - Low (0-0.34): Conventional, prefers routines, resistant to change.

    Conscientiousness:
    - High (0.75-1.0): Organized, disciplined, goal-oriented.
    - Medium (0.35-0.74): Moderately reliable, adaptable.
    - Low (0-0.34): Spontaneous, disorganized, impulsive.

    Extraversion :
    - High (0.75-1.0): Outgoing, sociable, energetic.
    - Medium (0.35-0.74): Balanced between sociability and introspection.
    - Low (0-0.34): Reserved, prefers solitude, introspective.
    
    Agreeableness:
    - High (0.75-1.0): Compassionate, cooperative, empathetic.
    - Medium (0.35-0.74): Sometimes cooperative, occasionally critical.
    - Low (0-0.34): Competitive, skeptical, values individualism.

    Neuroticism :
    - High (0.75-1.0): Emotionally sensitive, prone to stress or anxiety.
    - Medium (0.35-0.74): Generally stable, occasional stress.
    - Low (0-0.34): Calm, resilient, emotionally stable.

    With all this in mind, let's return a personality summary detailing everything in the following format:
    ###personality summary###
    """)

    #Task2
    tasks.append(
        """**Foundation: Understand the Core Traits**
    - Begin by analyzing the raw data provided (e.g., personality scores, demographic details, career, and life circumstances).
    - Create and answer questions reflecting on how these ALL THIS traits, scores, and circumstances interact to form the foundations of this person’s character. For example:
        How does their **Agreeableness** shape their interactions and relationships?
        How does  **Extraversion** affect their social habits and energy levels?
        How do their religious beliefs  influence their values and daily routines?
    - Explore possible contradictions or nuances. For example:
        How do their political beliefs coexist with their open-mindedness (high Openness score)?
    -These are just examples, but think and answer other questions that may be interesting to fully understand this profile.
    """)


    #Task3
    tasks.append(
        """**Inner World: Build Emotional and Psychological Depth**
    - Imagine their emotional landscape. What might they feel about their successes, failures, and relationships? For instance:
        Are they proud of their career success, or do they feel a sense of loneliness from prioritizing work over relationships?
        How do they manage stress or moments of self-doubt (given a moderate Neuroticism score)?
    - Reflect on their internal motivations and fears:
        What drives them to excel in their career? Is it passion, financial security, or something else?
        What fears might linger in their mind? (e.g., fear of growing old alone, fear of professional stagnation).
        -These are just examples, but think and answer other questions that may be interesting to fully understand this profile.
    """)


    #Task4
    tasks.append(
        """**External World: Connect Personality to Life Circumstances**
    - Consider how their traits manifest in their daily life, work, and relationships:
        How does their Conscientiousness influence their career?
        How does their Extraversion affect their friendships, romantic life, and workplace dynamics?
        What kind of leader, colleague, or friend might they be?
    - Reflect on their social and cultural context:
        How might their upbringing environment shape their worldview?
        How do their income and education levels affect their social class and interactions with others?
        -These are just examples, but think and answer other questions that may be interesting to fully understand this profile.
    """)


    #Task5
    tasks.append(
        """**Life Story: Imagine Their Past and Aspirations**
        Create a history for this person:
    - Reflect on their past: What events might have shaped their personality, beliefs, and current life?
        Did they grow up in a close-knit family, or did they have to work hard to achieve their success?
        Were there specific formative moments, such as academic achievements, career milestones, or personal losses?
    - Consider their future: What dreams and aspirations drive them forward? What challenges or obstacles do they face?
        Are they content with their current path, or do they secretly yearn for a change?
        -These are just examples, but think and answer other questions that may be interesting to fully understand this profile.
    
    """)


    #Task6
    tasks.append(
        """**Contradictions and Complexity**
    - Humans are not one-dimensional. Reflect on any contradictions in their personality or life:
        - For example, their  Openness might sometimes conflict with their structured and disciplined personality ( Conscientiousness).
        - How might their political beliefs  sometimes clash with their creative or open-minded tendencies?
    - Explore how these contradictions create complexity in their character and daily life.
    
    """)


    #Task7
    tasks.append(
        """ **Summary Generation**:
    - Generate a concise and insightful summary that captures the essence of the individual. This description should include economic profile and job information and must not exceed 200 characters. Enclose the summary within double ampersands like so:
        &&&& Summary &&&&
    """)


    #Task8
    tasks.append(
        """**Validation and Enrichment**:
        I don't want you to give me a filled-in structure, that will be done later, let's just reflect on what we should change in the next steps.
        
        Reflexion about:
    - Any values marked as None with specific, plausible information.
    - Vague or generic terms like "other", "low common", or "strange". Can we give more precise and descriptive alternatives?
    - Identify and correct any inconsistencies. For example, if the individual is under 12 years old, remove or simplify advanced concepts such as philosophical beliefs or complex life goals.
    - Alignment between the individual’s profession, income, education, and social class.
    - Social class based on income, profession, and overall life circumstances, and related fields accordingly.
    """)

    

    # === Generation Loop ===

    with open(PROFILES_JSON_PATH, 'a') as file:
        file.write('[')

    with open(REFLECTIONS_JSON_PATH, 'a') as file:
        file.write('[')


    for i in range(NUMBER_OF_PROFILES):
        pass

        system_prompt="You are an assistant who has to complete a series of tasks based on the following synthetic profiles of people living in Cambridge, Massachusetts (USA):"
        generated_taxonomy=fill_taxonomy()
        system_prompt=system_prompt+generated_taxonomy
        messages = [{"role": "system", "content":system_prompt}]


        reflections_dict = {}
        j=0
        for task in tasks:

            messages.append({"role": "user", "content": task})
            response = requests.post(URL, json={"model": MODEL, "messages": messages, "stream": False})
            if response.status_code == 200:
                
                reply = json.loads(response.text)
                message=reply['message']
                messages.append(message)
                dict_key='task_'+ str(j)
                reflections_dict[dict_key]=message['content']
                j+=1
                
                print("------------------------------------------------------------")
                print(f"Assistant: {message['content']}")
            else:
                print(f"Error {response.status_code}: {response.reason}")


        final_answer=-1
        count_problems=0
        while final_answer==-1:
            #Final response
            promt="""
                **Fill in the JSON Maintaining Original Structure **
            - Provide the filled profile ensuring it is cohesive, realistic, and have meaningful details. The final profile must adhere to correct JSON format.
                - You will have to fill in any value that does not have data, being consistent with the rest!!
                - You have already sent me the profile overview and the big 5 personality overview in the conversation, use this data to complete it.
            Final Notes:
            - You have to fill all the information!! None or null is not an accepted value
            - The JSON structure has to have all the keys and values structure that are provided to you in the system prompt with the same structure
            - Ensure the profile aligns with the provided data but also includes inferred details to make it more realistic.
            - Avoid generic descriptions. Instead, provide specific, human-like details that make this individual unique.
            - Think of this task as creating the blueprint for a character in a novel or movie. Every detail should feel deliberate and interconnected.
            - This is the initial profile again, you MUST mantain its structure and keys, and fill in with your information:
            """
            promt+=generated_taxonomy

            messages.append({"role": "user", "content": promt})
            response = requests.post(URL, json={"model": MODEL, "messages": messages, "format": "json", "stream": False})
            if response.status_code == 200:
                reply = json.loads(response.text)
                message=reply['message']
                messages.append(message)
                print("------------------------------------------------------------")
                try:
                    final_answer_dict = json.loads(message['content'])
                    final_answer_dict['id'] = i + 1 
                    print(message['content'])

                    #if has_empty_values(final_answer_dict): #Checking the dict is full
                    #    final_answer=-1
                    #else:
                    final_answer=0
                except Exception as e:
                    print(f"Error: {e}")  
                    final_answer=-1   


            else:
                print(f"Error {response.status_code}: {response.reason}")
                final_answer=-1

        print(final_answer_dict)
        reflections_dict[dict_key]=final_answer_dict

        with open(PROFILES_JSON_PATH, 'a') as file:
                    json.dump(final_answer_dict, file)
                    file.write(',\n')

        with open(REFLECTIONS_JSON_PATH, 'a') as file:
                    json.dump(reflections_dict, file)
                    file.write(',\n')

        print(f"Perfil {i+1} saved.")

    with open(PROFILES_JSON_PATH, 'a') as file:
                file.write(']')

    with open(REFLECTIONS_JSON_PATH, 'a') as file:
                file.write(']')



if __name__ == "__main__":
    main()