"""
profile_utils.py

Helper functions and base schema for generating synthetic human profiles 
based on demographic and psychological attributes from cambridge data.

This module includes:
- The base taxonomy structure (profile template)
- Data generation utilities (age, gender, education, personality, etc.)
- Validation helpers to check profile consistency

Author: José Miguel Nicolás García
"""

import json
import random
import numpy as np
import pickle
from faker import Faker
from scipy.stats import multivariate_normal


# ------------------------------------------------------------------------
# Base Taxonomy Definition
# ------------------------------------------------------------------------

#: `base_taxonomy` is a nested dictionary that represents the structure of a synthetic human profile.
#: It is designed to capture a wide range of human attributes across multiple dimensions,
#: including demographic, psychological, cognitive, behavioral, physical, and social data.
#:
#: Each field is initially set to `None` and is filled during the profile generation process.
#:
#: The structure supports hierarchical categories 


base_taxonomy = {
    "General": {  
        "General Description": None,  
        "Name": None,  
        "Age": None,
        "Education": None,
        "Place of Residence": "Cambridge Massachusetts (USA)",
    },

    "Identity": { 
        "Nationality": None,  
        "Sexual Orientation": None,  
        "Gender": None,  
        "Religious Beliefs": None,  
        "Political Ideology": None,  
    },

    "Profession": {
        "Industry":None,
        "Industry Mean salary":None,
        "Personal Salary":None,
        "Job":None,
    },

    "Psychological and Cognitive": { 
        "Personality/Big Five Traits":{
            "Agreeable": None,
            "Extraversion": None,
            "Openness": None,
            "Conscientiousness": None,
            "Neuroticism": None,
            "General Big Five Description": None,
            
        },
       "Cognitive": {
        "Emotional Intelligence": None,  
        "Logical-Mathematical": None,  
        "Creativity": None,  
        "Social Intelligence": None,  
        "Self-Awareness": None,  
       },
       "Emotional": {
        
       },
       "Motivations": { 
            "Goals and Aspirations": None,  
            "Career Aspirations": None, 
            "Life Goals": None,  
        },
        "Strengths": None,
        "Weaknesses": None
    },

    "Behavioral": { 
        "Social": {
            "Relationships": { 
                "Family": { 
                    "Family Background": None,   
                    "Parental Relationships": None,  
                },
                "Friends": {  
                    "Social Networks": None, 
                    "Social Skills": None,  
                },
                "Marital Status": None,   
                "Workplace Relationships": None,  
        },
       
        "Social Roles and Status": {  
            "Role in Community": None, 
            "Social Class": None,  
        },
        },

        "Habits and Routines": {
            "Daily Routine": None, 
            "Leisure Activities": None,  
            "Work Habits": None,  
            "Health Habits": None,  
            "Social Habits": None, 
        },
    },

    "Physical/Biological/Movility": {  
        "Physiological": {

        },
        "Anatomical": {
            "Hair Color": None,
            "Eye Color": None,
            "Complexion": None,
            "Height": None,  
            "Weight": None,
            "Overweight":None, 
        },
        "Health": {
            "Health Status": None,
            "Disabilities": None
        },
    },    
}



def generate_age_and_gender() -> tuple[int, str]:
    """
    Generate a realistic age and gender using Cambridge, MA population pyramid data.

    The age is sampled from actual census age brackets. Gender is chosen probabilistically
    based on total male/female population proportions.

    Returns:
        tuple[int, str]: A tuple with:
            - age (int): A randomly selected age between 0 and 100
            - gender (str): "Male" or "Female"

    References:
        - U.S. Census Bureau, ACSST1Y2022.S0101 (Cambridge City, MA)
          https://data.census.gov/table/ACSST1Y2022.S0101?g=160XX00US2511000
    """

    age_brackets = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 24), (25, 29), (30, 34), (35, 39),
                (40, 44), (45, 49), (50, 54), (55, 59), (60, 64), (65, 69), (70, 74), (75, 79), 
                (80, 84), (85, 100)]  
    male_population = [2941, 2053, 1765, 4131, 8351, 9733, 7203, 4754, 3187, 2637, 2164, 2126, 2140, 1989, 1773, 968, 627, 591]
    female_population = [2001, 1720, 1598, 4690, 8715, 9031, 6091, 4808, 2864, 2619, 2383, 2109, 2128, 2596, 2169, 1448, 843, 1016]

    total_population = sum(male_population) + sum(female_population)
    male_probabilities = [pop / total_population for pop in male_population]
    female_probabilities = [pop / total_population for pop in female_population]

    male_probabilities /= np.sum(male_probabilities)
    female_probabilities /= np.sum(female_probabilities)

    gender = 'Male' if np.random.rand() < sum(male_population) / total_population else 'Female'
    if gender == 'Male':
        selected_index = np.random.choice(range(len(age_brackets)), p=male_probabilities)
    else:
        selected_index = np.random.choice(range(len(age_brackets)), p=female_probabilities)
    
    min_age, max_age = age_brackets[selected_index]
    exact_age = np.random.randint(min_age, max_age + 1)
    
    return exact_age, gender



def generate_ethnicity() -> str:
    """
    Sample a realistic ethnic identity based on self-reported racial combinations
    from the 2020 Census data for Cambridge, MA.

    Returns:
        str: A string describing the selected ethnicity.

    References:
        - U.S. Census Bureau, DECENNIALDHC2020.P8
          https://data.census.gov/table/DECENNIALDHC2020.P8?g=160XX00US2511000
    """
    ethnic_groups = {
        "White": 67887,
        "Black or African American": 12520,
        "American Indian and Alaska Native": 290,
        "Asian": 22743,
        "Native Hawaiian and Other Pacific Islander": 54,
        "Some Other Race": 4036,
        "White and  Black or African American": 1320,
        "White and  American Indian and Alaska Native": 443,
        "White and  Asian": 2650,
        "White and  Native Hawaiian and Other Pacific Islander": 33,
        "White and  Some Other Race": 4484,
        "Black or African American and  American Indian and Alaska Native": 88,
        "Black or African American and  Asian": 125,
        "Black or African American and  Native Hawaiian and Other Pacific Islander": 27,
        "Black or African American and  Some Other Race": 668,
        "American Indian and Alaska Native and  Asian": 34,
        "American Indian and Alaska Native and  Native Hawaiian and Other Pacific Islander": 0,
        "American Indian and Alaska Native and  Some Other Race": 55,
        "Asian and  Native Hawaiian and Other Pacific Islander": 69,
        "Asian and  Some Other Race": 73,
        "Native Hawaiian and Other Pacific Islander and  Some Other Race": 12
    }

    ethnicity_names = list(ethnic_groups.keys())
    populations = list(ethnic_groups.values())
    total_population = sum(populations)
    probabilities = [pop / total_population for pop in populations]

    return np.random.choice(ethnicity_names, p=probabilities)




def generate_income() -> tuple[int, str]:
    """
    Generate a realistic annual personal income based on 2023 ACS household income distribution
    for Cambridge, Massachusetts (USA).

    This function samples an income bracket according to official census percentages,
    then randomly selects a value within that range.

    Returns:
        tuple[int, str]: 
            - exact_income (int): Randomly generated value in USD
            - income_range (str): Label for the income bracket (e.g., "75,000 to 99,999")

    References:
        U.S. Census Bureau (2023). ACSST1Y2023.S1901: Income in the Past 12 Months.
        https://data.census.gov/table/ACSST1Y2023.S1901?g=160XX00US2511000
    """
    income_ranges = [
        "<10,000",
        "10,000 to 14,999",
        "15,000 to 24,999",
        "25,000 to 34,999",
        "35,000 to 49,999",
        "50,000 to 74,999",
        "75,000 to 99,999",
        "100,000 to 149,999",
        "150,000 to 199,999",
        "200,000 or more"
    ]

    income_limits = [
        (0, 10000),
        (10000, 14999),
        (15000, 24999),
        (25000, 34999),
        (35000, 49999),
        (50000, 74999),
        (75000, 99999),
        (100000, 149999),
        (150000, 199999),
        (200000, 300000)
    ]

    household_probabilities = [4.2, 2.2, 4.3, 2.2, 5.4, 9.7, 9.2, 18.4, 14.4, 30.1]
    household_probabilities = [prob / sum(household_probabilities) for prob in household_probabilities]
    selected_range_index = np.random.choice(len(income_ranges), p=household_probabilities)
    selected_range_name = income_ranges[selected_range_index]
    range_min, range_max = income_limits[selected_range_index]
    if selected_range_name == "200,000 or more":
        exact_income = random.randint(range_min, range_max)
    else:
        exact_income = random.randint(range_min, range_max)  
    return exact_income, selected_range_name


def generate_lgbt_identity() -> str:
    """
    Generate a random LGBT+ identity according to national survey proportions.

    Returns:
        str: A randomly selected LGBT+ label, such as "Bisexual" or "Pansexual".

    References:
        Gallup (2024). LGBT Identification in the U.S.
        https://news.gallup.com/poll/611864/lgbtq-identification.aspx
    """
    identities = ['Homosexual', 'Bisexual', 'Transgender', 'Pansexual', 'Asexual', 'Queer', 'Other LGBT+']
    probabilities = [33.2, 57.3, 11.8, 1.7, 1.3, 0.1, 1.1]  
    probabilities = [float(i)/sum(probabilities) for i in probabilities]  
    return np.random.choice(identities, p=probabilities)


def generate_sexual_orientation_and_genere(age: int) -> str:
    """
    Determine sexual orientation based on age-specific LGBT+ identification rates.

    Age groups are mapped to corresponding likelihoods of identifying as LGBT+.
    If the individual falls within a group with a non-zero probability, they may
    be assigned a specific identity using `generate_lgbt_identity`.

    Args:
        age (int): Age of the individual.

    Returns:
        str: "Heterosexual, CIS" or a specific LGBT+ identity.

    References:
        Boston Indicators
        https://www.bostonindicators.org/reports/report-website-pages/lgbt-report/demographic-overview
    """
    age_groups = {
        "0-24": 15.5,
        "25-34": 10.4,
        "35-44": 6.0,
        "45-54": 5.7,
        "55-64": 5.3,
        "65-100": 2.7
    }

    for age_range, probability in age_groups.items():
        age_min, age_max = map(int, age_range.split('-'))
        if age_min <= age <= age_max:
            is_lgbt = np.random.rand() < (probability / 100)  
            if is_lgbt:
                return generate_lgbt_identity()  
            else:
                return "Heterosexual, CIS"
    return "Heterosexual, CIS"

def generate_USBorn() -> bool:
    """
    Simulate whether an individual was born in the United States based on
    real proportions for the Cambridge (MA) population.

    Returns:
        bool: True if the individual was born in the U.S., False if foreign-born.

    References:
        U.S. Census Bureau (2023). 
        https://data.census.gov/table/ACSDP1Y2023.DP02?g=160XX00US2511000
    """
    citizenship_status = [True, False]
    probabilities = [38.5, 61.5]
    probabilities = [p / sum(probabilities) for p in probabilities]  
    return np.random.choice(citizenship_status, p=probabilities)



def generate_country(country_distribution_path: str = "data/cambridge_countries.json") -> str:
    """
    Select a country of birth based on actual foreign-born population data 
    from Cambridge, MA. If the individual is U.S.-born, returns 'United States'.

    Args:
        country_distribution_path (str): Path to JSON with foreign-born data
            from https://datausa.io API (must include 2022 entries and 3-letter codes).

    Returns:
        str: Country of origin.

    References:
        DataUSA.io. Birthplace Distribution in Cambridge, MA.
        https://datausa.io/profile/geo/cambridge-ma/

        json
        http://datausa.io/api/data?Geography=16000US2511000&Nativity=2&measure=Total%20Population,Total%20Population%20MOE%20Appx&drilldowns=Birthplace&properties=Country%20Code
    """
    if generate_USBorn():
        return "United States"

    with open(country_distribution_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid_data = [
        entry for entry in data["data"]
        if entry["Year"] == "2022"
        and isinstance(entry.get("Country Code"), str)
        and len(entry["Country Code"]) == 3
        and entry["Country Code"].isalpha()
    ]

    total = sum(entry["Total Population"] for entry in valid_data)
    countries = [entry["Birthplace"] for entry in valid_data]
    weights = [entry["Total Population"] / total for entry in valid_data]

    return random.choices(countries, weights=weights, k=1)[0]


def generate_education(age: int) -> str | None:
    """
    Determine an individual's current education level based on age group
    and U.S. enrollment rates. Returns the most likely education category.

    Args:
        age (int): Age of the individual.

    Returns:
        str | None: Education level or None if not yet enrolled.

    References:
        U.S. Census Bureau (2023). Educational Attainment.
        https://data.census.gov/table/ACSST1Y2023.S1501?g=160XX00US2511000
    """
    enrollment_data = [
        {"age_range": (3, 4), "enrollment_rate": 74.8, "level": "Nursery school, preschool"},
        {"age_range": (5, 9), "enrollment_rate": 96.6, "level": "Kindergarten"},
        {"age_range": (10, 17), "enrollment_rate": 100.0, "level": "Kindergarten"},
        {"age_range": (18, 19), "enrollment_rate": 99.0, "level": "Atending College, Undergraduate"},
        {"age_range": (20, 22), "enrollment_rate": 69.7, "level": "Atending College, Undergraduate"}
    ]

    for entry in enrollment_data:
        if entry["age_range"][0] <= age <= entry["age_range"][1]:
            if np.random.random() < (entry["rate"] / 100):
                return entry["level"]

    if age >= 22:
        levels = [
            "High school or equivalent degree",
            "Some college, no degree",
            "Associate's degree",
            "Bachelor's degree",
            "Graduate or professional degree"
        ]
        probabilities = [6.0, 5.3, 3.0, 27.9, 54.8]
        probabilities = [p / sum(probabilities) for p in probabilities]
        return np.random.choice(levels, p=probabilities)

    return None



def generate_disabilities(age: int) -> list[str]:
    """
    Simulate the presence of one or more disabilities for an individual 
    based on U.S. Census age-specific probabilities.

    Args:
        age (int): Age of the individual.

    Returns:
        list[str]: List of detected disabilities or ["No disability"].

    References:
        U.S. Census Bureau (2023). Disability Characteristics.
        https://data.census.gov/table/ACSST1Y2023.S1810?g=160XX00US2511000
    """
    probabilities = {
        "hearing": [0.7, 1.1, 0.7, 0.4, 1.3, 10.3, 19.5], #These are porcentages (0.7% an go ong)
        "vision": [0, 0, 0.9, 0.5, 1.6, 2.3, 0],
        "cognitive": [3.3, 0, 5.2, 4.5, 6.4, 6.8, 10.3],
        "ambulatory": [0, 0, 2.3, 1.4, 3.8, 12.0, 20.1],
        "self_care": [0, 0, 0.5, 0.2, 0.8, 5.1, 9.0],
        "independent_living": [0, 0, 1.8, 1.2, 2.8, 11.3, 21.9]
    }

    age_ranges = [(0, 5), (5, 18), (18, 34), (35, 64), (65, 74), (75, 120)]

    for i, (low, high) in enumerate(age_ranges):
        if low <= age < high:
            disabilities = [key for key, probs in probabilities.items() if random.random() < probs[i] / 100]
            return disabilities or ["No disability"]


def generate_big5profiles(
    age: int,
    sex: str,
    country: str,
    model_path: str = "data/big_five_parameters.pkl"
) -> dict[str, float]:
    """
    Generate a Big Five personality profile using pre-trained regression models 
    and a multivariate Gaussian distribution of traits.

    Args:
        age (int): Age of the individual.
        gender (str): Gender ("Male" or "Female").
        country (str): Country of origin (e.g., "United States").
        model_path (str): Path to the .pkl file containing trained models and covariance.

    Returns:
        dict[str, float]: Dictionary with personality traits:
            - agreeable_score
            - extraversion_score
            - openness_score
            - conscientiousness_score
            - neuroticism_score

    References:
        Big Five Dataset: https://ipip.ori.org/
        Dataset: 300,000+ user submissions with rated personality scores.
    """
    traits = ["agreeable_score", "extraversion_score", "openness_score", "conscientiousness_score", "neuroticism_score"]

    USA = 1 if country == "United States" else 0
    sex = 1 if sex == "Male" else 2


    # Load models and covariance matrix
    with open("./data/big_five_parameters.pkl", "rb") as f:
        data = pickle.load(f)
        
    regression_models = data["models"]
    covariance_matrix = data["covariance_matrix"]
    scaler=data["scaler"]


    # Prepare the input and normalize it
    X_input = np.array([[age, sex, USA]])
    X_input_scaled = scaler.transform(X_input)

    # Compute conditional means using regression models
    conditional_means = [regression_models[trait].predict(X_input_scaled)[0] for trait in traits]

    # Generate a single sample from the multivariate normal distribution
    generated_data = multivariate_normal.rvs(mean=conditional_means, cov=covariance_matrix, size=1)

    profile = {trait: value for trait, value in zip(traits, generated_data)}

    return profile


def generate_height_weight_overweight(gender: str) -> tuple[float, float, bool]:
    """
    Simulate height, weight, and overweight status using BMI distribution
    for adults in the U.S., adjusted by gender.

    Args:
        gender (str): "Male", "Female", or "Other".

    Returns:
        tuple[float, float, bool]: 
            - height (cm), weight (kg), is_overweight (BMI >= 25)

    References:
        World Population Review (2024). Average Height by State.
        https://worldpopulationreview.com/state-rankings/average-height-by-state
    """
    if gender == "Male":
        height = round(random.gauss(175.26 , 10), 1)  
        bmi = random.gauss(22.5, 3)  
    elif gender == "Female":
        height = round(random.gauss(162.56, 8), 1)  
        bmi = random.gauss(22.5, 3)  
    else:
        height = round(random.gauss(168, 10), 1)  
        bmi = random.gauss(22.5, 3)  

    weight = round((bmi * (height / 100) ** 2), 1)
    overweight = bmi >= 25  
    return height, weight, overweight


def choose_political_ideology() -> str:
    """
    Select a political ideology label based on 2020 election data 
    from Cambridge, MA.

    Returns:
        str: One of: "Democratic", "Republican", "Libertarian", "Green", "Other"

    References:
        DataUSA.io - Presidential Elections in Massachusetts.
        https://datausa.io/profile/geo/cambridge-ma/
    """
    ideologies = ["Democratic", "Republican", "Libertarian", "Green", "Other"]
    probabilities = [61.6, 31.9, 1.29, 0.51, 1.17]
    return random.choices(ideologies, probabilities)[0]


def choose_religion() -> str:
    """
    Choose a religion based on Pew Research Center data for the Boston metro area.

    Returns:
        str: A religion or belief system.

    References:
        Pew Research Center. Religious Landscape Study - Boston Metro Area.
        https://www.pewresearch.org/religious-landscape-study/database/metro-area/boston-metro-area/
    """
    religions = [
        "Evangelical Protestant", "Mainline Protestant", "Catholic", "Mormon", 
        "Orthodox Christian", "Jehovah's Witness", "Other Christian", 
        "Jewish", "Muslim", "Buddhist", "Hindu", "Other World Religions", 
        "Other Faiths", "Atheist", "Agnostic", "Nothing in particular", "Don't know"
    ]
    
    probabilities = [
        9, 13, 29, 1, 2, 1, 1,  # Christian denominations
        4, 1, 1, 1, 1, 2,     # Non-Christian Faiths
        4, 9, 20, 1              # Unaffiliated
    ]
    
    return random.choices(religions, probabilities)[0]

def choose_industry_and_salary(age: int) -> tuple[str, int, str | int]:
    """
    Simulate industry sector, industry average salary, and individual's salary
    or unemployment status based on ACS data for Cambridge.

    Args:
        age (int): Age of the individual.

    Returns:
        tuple[str, int, str | int]: 
            - industry (str)
            - industry_mean_salary (int)
            - personal_salary (int or "Unemployed")

    References:
        U.S. Census Bureau / DataUSA (2022). Cambridge Employment & Income.
        https://datausa.io/profile/geo/cambridge-ma/
    """
    if age < 16:
        return "Not in labor force", "Not in labor force", "Not in labor force"

   
    industries = [
        "Agriculture, Forestry",
        "Mining, Quarrying, & Oil",
        "Construction",
        "Manufacturing",
        "Wholesale Trade",
        "Retail Trade",
        "Transportation",
        "Utilities",
        "Information",
        "Finance & Insurance",
        "Real Estate & Rental",
        "Professional, Scientific, & Technical Services",
        "Management",
        "Administrative Services",
        "Educational Services",
        "Health Care & Social Assistance",
        "Arts, Entertainment, & Recreation",
        "Accommodation & Food Services",
        "Others: (Extrange industry)"
    ]
    
  
    workers = [161, 5, 932, 4840, 474, 3690, 127000, 321, 2380, 3070, 929, 16700, 57, 111, 195000, 8220, 133000, 1920, 2200]
    salaries = [232000, 29000, 142000, 212000, 156000, 74700, 86600, 204000, 223000, 216000, 159000, 212000, 209000, 95900, 100000, 129000, 56000, 38300, 66000]
    
    total_workers = sum(workers)
    probabilities = [w / total_workers for w in workers]
    selected_industry = random.choices(industries, weights=probabilities, k=1)[0]
    
    industry_index = industries.index(selected_industry)
    medium_salary = salaries[industry_index]
    generated_salary =  int(random.gauss(medium_salary, 0.25 * medium_salary))
    

    employment_rate = 0.671  
    is_employed = random.random() < employment_rate  #https://data.census.gov/profile/Cambridge_city,_Massachusetts?g=160XX00US2511000

    if not is_employed:
        return selected_industry, medium_salary, "Unemployed"
    
    
    return selected_industry, medium_salary ,generated_salary


def decide_marital_status(sex: str) -> str:
    """
    Decide marital status based on age, using U.S. Census ACS probabilities.

    Args:
        sex (str): Sex of the individual.

    Returns:
        str: One of "Never married", "Married", "Divorced", "Widowed", "Separated"

    References:
        U.S. Census Bureau (2022). 
        https://data.census.gov/profile/Cambridge_city,_Massachusetts?g=160XX00US2511000#families-and-living-arrangements
    """
    statuses = ["Married, not separated", "Widowed", "Divorced", "Separated", "Never Married"]
    probabilities = [34.0, 1.1, 4.2, 0.5, 60.1] if sex == "Male" else [31.3, 4.0, 7.5, 1.1, 56.2]
    probabilities = [p / sum(probabilities) for p in probabilities]
    return np.random.choice(statuses, p=probabilities)


def has_empty_values(data: dict) -> bool:
    """
    Recursively check whether any value in a nested dictionary is None or empty.

    Args:
        data (dict): Profile dictionary to check.

    Returns:
        bool: True if any empty or missing value is found, False otherwise.
    """
    for key, value in data.items():
        if value is None or value == "":  
            return True
        if isinstance(value, dict):  
            if has_empty_values(value):
                return True
        elif isinstance(value, list):  
            for item in value:
                if isinstance(item, dict):
                    if has_empty_values(item):
                        return True
                elif item is None or item == "":
                    return True
    return False


def has_required_fields(base: dict, generated: dict) -> bool:
    """
    Check that all specified fields exist and are non-empty in a nested profile.

    Args:
        base (dict): Reference dictionary.
        generated (dict): Full profile dictionary.


    Returns:
        bool: True if all fields are present and filled, False if any are missing/empty.
    """
    for key, value in base.items():
        if key not in generated:  # Key missing in the generated JSON
            print("Missing key:")
            print(key)
            return False
        if isinstance(value, dict):  # If the value is a dictionary, check recursively
            if not isinstance(generated[key], dict):  # Type mismatch
                return False
            if not has_required_fields(value, generated[key]):  # Recurse into the nested dictionary
                return False
        elif isinstance(value, list):  # If the value is a list
            #if not isinstance(generated[key], list):  # Type mismatch
                #return False
            for base_item in value:
                if isinstance(base_item, dict):
                    if not any(
                        has_required_fields(base_item, gen_item)
                        for gen_item in generated[key]
                        if isinstance(gen_item, dict)
                    ):
                        return False
        # For other types, no further checking is needed as existence suffices.
    return True



def has_exact_keys(base: dict, generated: dict) -> bool:
    """
    Check that a nested dictionary has the exact same structure (keys and subkeys)
    as a reference dictionary. This is useful to ensure schema consistency.

    Args:
        base (dict): Reference dictionary.
        generated (dict): Full profile dictionary.

    Returns:
        bool: True if the structure matches exactly, False otherwise.
    """
    base=base_taxonomy

    if isinstance(base, dict):
        # Both must be dictionaries with the same keys
        if not isinstance(generated, dict) or set(base.keys()) != set(generated.keys()):
            return False
        # Recursively check the structure of each key
        return all(has_exact_keys(base[key], generated[key]) for key in base)
    elif isinstance(base, list):
        # Both must be lists
        if not isinstance(generated, list):
            return False
        # Check the structure of the first item if it's a list of objects
        if len(base) > 0 and isinstance(base[0], dict):
            return all(has_exact_keys(base[0], item) for item in generated if isinstance(item, dict))
        return True  # Otherwise, assume lists of primitives don't need further validation
    else:
        # For primitive types, we just ensure the key exists
        return True


def fill_taxonomy() -> dict:
    """
    Construct a complete synthetic human profile by filling a predefined
    taxonomy structure using dedicated attribute generators.

    This function creates a copy of the base taxonomy and populates its fields 
    with realistic demographic, psychological, and physical attributes. It uses 
    predefined generators for each attribute (e.g., age, gender, education, 
    salary, personality, etc.), sampled according to real-world distributions 
    from sources like the U.S. Census, Pew Research Center, and Gallup.

    Returns:
        dict: A nested dictionary representing a complete synthetic profile,
                compatible with the `base_taxonomy` schema.

    Notes:
        - This function integrates all attribute generation pipelines in a fixed order.
        - Profile is returned both as a dictionary and printed in formatted JSON.
        - If you wish to validate the completeness of the output, use helper functions 
            such as `has_empty_values()` or `has_required_fields()`.

    """
    fake = Faker(['en_US'])
    
    age,gender=generate_age_and_gender()
    name = fake.name_male() if gender == 'Male' else fake.name_female()
    sexual_orientation=generate_sexual_orientation_and_genere(age)
    education= generate_education(age)
    disabilities=generate_disabilities(age)
    ideology=choose_political_ideology()
    height, weight, overweight= generate_height_weight_overweight(gender)
    country = generate_country()
    religion= choose_religion()
    industry, industry_mean_salary, salary=choose_industry_and_salary(age)
    marital_status=decide_marital_status(gender)
    big_five=generate_big5profiles(age,gender,country)
    



    # Copy of base dictionary
    profile = base_taxonomy.copy()

    profile["General"]["Name"] = name
    profile["General"]["Age"] = age
    profile["General"]["Education"] = education

    profile["Identity"]["Nationality"] = country
    profile["Identity"]["Sexual Orientation"] = sexual_orientation
    profile["Identity"]["Gender"] = gender
    profile["Identity"]["Religious Beliefs"] = religion
    profile["Identity"]["Political Ideology"] = ideology

    profile["Profession"]["Industry"] = industry
    profile["Profession"]["Industry Mean salary"] = industry_mean_salary
    profile["Profession"]["Personal Salary"] = salary


    profile["Psychological and Cognitive"]["Personality/Big Five Traits"]["Agreeable"] = big_five["agreeable_score"]
    profile["Psychological and Cognitive"]["Personality/Big Five Traits"]["Extraversion"] = big_five["extraversion_score"]
    profile["Psychological and Cognitive"]["Personality/Big Five Traits"]["Openness"] = big_five["openness_score"]
    profile["Psychological and Cognitive"]["Personality/Big Five Traits"]["Conscientiousness"] = big_five["conscientiousness_score"]
    profile["Psychological and Cognitive"]["Personality/Big Five Traits"]["Neuroticism"] = big_five["neuroticism_score"]


    profile["Behavioral"]["Social"]["Relationships"]["Marital Status"] = marital_status

    profile["Physical/Biological/Movility"]["Anatomical"]["Height"] = f"{height} cm"
    profile["Physical/Biological/Movility"]["Anatomical"]["Weight"] = f"{weight} kg"
    profile["Physical/Biological/Movility"]["Anatomical"]["Overweight"] = "Yes" if overweight else "No"


    profile["Physical/Biological/Movility"]["Health"]["Disabilities"] = disabilities

    print(json.dumps(profile, indent=4))
    profile=json.dumps(profile)

    return profile
