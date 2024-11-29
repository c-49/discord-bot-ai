import discord
from discord.ext import commands
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix='!', intents=intents)

model_name = 'gpt2-medium'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(prompt):
    full_prompt = f"Q: {prompt}\nA:"
    input_ids = tokenizer.encode(full_prompt, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
    output = model.generate(
        input_ids, 
        max_length=input_ids.shape[1] + 50,
        num_return_sequences=1, 
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_p=0.9,
        no_repeat_ngram_size=2,
        do_sample=True
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    ai_response = response.split("A:")[-1].strip()
    
    # Post-processing to ensure we're only getting the relevant part of the response
    ai_response = ai_response.split('\n')[0]  # Take only the first line
    
    return ai_response if ai_response else "I'm not sure how to respond to that."

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.command(name='ask')
async def ask(ctx, *, question):
    response = generate_response(question)
    await ctx.send(response)

bot.run('') # discord api key