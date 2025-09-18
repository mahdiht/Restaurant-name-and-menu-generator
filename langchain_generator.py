from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain


llm = Ollama(model="llama3.1", temperature=0.9)

def generate_restaurant_name_and_items(cuisine):
    prompt_template_name = PromptTemplate(
        input_variables =['cuisine'],
        template = "I want to open a restaurant for {cuisine} food. Suggest only one fency name for this. Response in maximum 3 words. Don't add any explanation."
    )

    name_chain =LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")

    prompt_template_items = PromptTemplate(
        input_variables = ['restaurant_name'],
        template="""Suggest some menu items for {restaurant_name}. response in comma separated format. Don't add any explanation."""
    )

    food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")

    chain = SequentialChain(chains = [name_chain, food_items_chain], 
                        input_variables=["cuisine"],
                        output_variables=["restaurant_name", "menu_items"]
                        )
    response = chain({"cuisine": cuisine})

    return response

if __name__ == "__main__":
    print(generate_restaurant_name_and_items("Italian"))