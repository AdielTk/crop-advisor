import warnings
warnings.filterwarnings('ignore')
import gradio as gr
from crewai import Agent,Task,Crew,LLM,Process
llm=LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434",
    temperature=0.2

) 
Disease_Identifier=Agent(
    role="Crop Disease Identifier",
    goal="Given a crop and its symptoms, identify up to 3 most likely "
    "diseases or pests,explain briefly why each fits, and state "
    "your confidence(high,medium or low) for each.",
    backstory="You are a plant pathologist with years of experience advising "
               "smallholder farmers in Ghana. "
               "You are careful and honest: if the syptoms are too vague or unclear to "
               "decide, you say you are uncertain instead of guessing. "
              "Your work is the basis for "
              "the Cause and Spread Specialist.",
    llm=llm,
    allow_delegation=False,
    verbose=True
)
Cause_Spread_Analyst=Agent(
    role="Cause and Spread Analyst",
    goal="For each disease identified,explain what causes it(virus,fungi, "
    "bacteria or pest) and in simple language how it spread from plant to plant and "
    "from farm to farm.",
    backstory="You are an agricultural scientist who have worked with many farmers in Ghana "
             "on crop  disease control. "
             "You explain things to them without using jargon,and you never invent "
             "facts:if you are unsure how a disease spreads,you say so. "
             "Your explanation would be used by the Treatment and Prevention Specialist.",
    llm=llm,
    allow_delegation=False,
    verbose=True         
)
Treatment_Prevention_Specialist=Agent(
    role="Treatment and Prevention Specialist",
       goal= "Using the disease and its causes, give practical control measures "
        "a smallholder farmer can follow: cultural practices, resistant "
        "varieties, and treatments, plus steps to prevent the disease"
        "returning next season",
    backstory= "You are an agronomist who has advised smallholder farmers in "
               "Ghana for many years. "
                "You prefer low-cost, practical solutions that farmers can "
               "actually afford and find locally. "
               "You never invent chemical names or give exact dosages. Instead, "
              "you advise the farmer to confirm any treatment with a Ministry "
             "of Food and Agriculture (MoFA) extension officer. "
              "If you are unsure, you say so.",
    llm=llm,
    allow_delegation=False,
    verbose=True
)
reviewer = Agent(
    role="Farmer Advice Reviewer",
    goal=(
    "Rewrite the findings from the other agents as short, clear advice "
    "a smallholder farmer can easily understand. Always keep the name "
    "of the most likely disease, its cause, and how it spreads. Remove "
    "only invented chemical names and dosages. If the earlier agents "
    "were uncertain, say so honestly instead of deleting the information."
    ),
    backstory=(
        "You are an agricultural extension communicator in Ghana who "
        "explains technical information to farmers in plain, simple "
        "English. "
        "You avoid jargon and keep sentences short. "
        "You are strict about safety: you remove or flag specific "
        "chemical names and exact dosages, and instead tell the farmer "
        "to confirm treatments with a Ministry of Food and Agriculture "
        "(MoFA) extension officer. "
        "If the earlier agents were uncertain, you say so honestly "
        "instead of sounding confident."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True
)
identify_task = Task(
    description="Identify the likely disease affecting {crop} with these symptoms: {symptoms}.If the symptoms are too vague,say you are uncertain",
    expected_output="A list of up to 3 possible diseases, each with a one-line reason and a confidence level.",
    agent=Disease_Identifier,
)
cause_spread_task = Task(
    description=(
        "Using the diseases identified for {crop} with these symptoms: "
        "{symptoms}, explain what causes each disease (virus, fungus, "
        "bacteria or pest) and how it spreads from plant to plant and "
        "farm to farm. Use simple language. If you are unsure how a "
        "disease spreads, say so instead of guessing."
    ),
    expected_output=(
        "For each disease: its cause, and a short list of how it spreads."
    ),
    agent=Cause_Spread_Analyst,
    context=[identify_task],
)

treatment_task = Task(
    description=(
        "Using the identified diseases and their causes for {crop}, "
        "give practical control and prevention measures a smallholder "
        "farmer in Ghana can follow. Cover cultural practices, resistant "
        "varieties, treatments, and steps to prevent the disease "
        "returning next season. Do not give exact dosages or invent "
        "chemical names. Advise the farmer to confirm any treatment "
        "with a MoFA extension officer."
    ),
    expected_output=(
        "Two short lists: (1) control measures for now, "
        "(2) prevention steps for the next season."
    ),
    agent=Treatment_Prevention_Specialist,
    context=[identify_task, cause_spread_task],
)
review_task = Task(
    description=(
        "Review the disease identification, cause and spread, and "
        "treatment advice for {crop} with these symptoms: {symptoms}. "
        "Rewrite everything as simple advice for a smallholder farmer. "
        "You MUST keep the disease name, its cause, and how it spreads "
        "from the earlier agents' work. Do not add new facts. Remove "
        "or flag only specific chemical names and dosages."
    ),
    expected_output=(
        "Advice in exactly this format:\n"
        "**What it is:** (the name of the most likely disease, your "
        "confidence, and its cause, in 1-2 sentences)\n"
        "**How it spreads:** (short list)\n"
        "**Signs:** (short list)\n"
        "**What to do:** (short list of practical steps)\n"
        "**When to get help:** (when to contact a MoFA extension officer)\n"
        "Include a note on any uncertainty."
    ),
    agent=reviewer,
    context=[identify_task, cause_spread_task, treatment_task],
)

crew = Crew(
    agents=[Disease_Identifier, Cause_Spread_Analyst,
            Treatment_Prevention_Specialist, reviewer],
    tasks=[identify_task, cause_spread_task, treatment_task, review_task],
    process=Process.sequential,
    verbose=True
)
def run_and_save(crop, symptoms, filename):
    result = crew.kickoff(inputs={"crop": crop, "symptoms": symptoms})

    names = ["Disease Identifier", "Cause and Spread Analyst",
             "Treatment and Prevention Specialist", "Reviewer (final answer)"]

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"INPUT: {crop} | {symptoms}\n\n")
        for name, task_output in zip(names, result.tasks_output):
            f.write(f"===== {name} =====\n")
            f.write(task_output.raw + "\n\n")

            import gradio as gr

def run_crew(crop, symptoms):
    crop = (crop or "").strip()
    symptoms = (symptoms or "").strip()

    # Input validation: don't waste a long run on unusable input
    if crop.lower() in ["", "unknown", "none", "n/a", "?"]:
        return "Please enter the name of the crop (for example: cassava, maize, cocoa)."
    if len(symptoms) < 10:
        return ("Please describe the symptoms in more detail. Mention the colour, "
                "shape and position of the marks, and which part of the plant is affected.")

    try:
        result = crew.kickoff(inputs={"crop": crop, "symptoms": symptoms})
        return str(result)
    except Exception as e:
        return ("Something went wrong. Check that Ollama is running and the "
                f"llama3.2 model is installed.\n\nDetails: {e}")

demo = gr.Interface(
    fn=run_crew,
    inputs=[
        gr.Textbox(label="Crop", placeholder="e.g. cassava"),
        gr.Textbox(label="Symptoms", placeholder="e.g. yellow patterns on leaves", lines=3),
    ],
    outputs=gr.Markdown(label="Advice"),
    title="Farmer Advisory Crew",
    description=(
        "A multi-agent prototype that suggests possible crop diseases and "
        "practical advice. It is a learning project, not a diagnosis. "
        "Always confirm with a MoFA extension officer. Answers can take a few "
        "minutes because the model runs on this computer."
    ),
    examples=[
        ["cassava", "yellow patterns on leaves"],
        ["cocoa", "dark rotting patches on pods"],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()