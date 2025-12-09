import os
import csv
import flask
import gdown
import discord
import asyncio
from dotenv import load_dotenv
from discord.ext import commands
from ctransformers import AutoModelForCausalLM
from webserver import b

load_dotenv()
TOKEN = os.getenv('TOKEN')

b()

# ======================
# Load model
# ======================

FILE_ID = "1rLgrof76RwL6AkArjmVRh7tDunV0s2ni"
MODEL_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

def download_model():
    if not os.path.exists(MODEL_FILE):
        print("📥 Downloading model from Google Drive... Please wait...")
        url = f"https://drive.google.com/uc?export=download&id={FILE_ID}&confirm=t"
        gdown.download(url, MODEL_FILE, quiet=False)
        print("✅ Model downloaded!")
    else:
        print("⚡ Model already exists. Skipping download.")


download_model()

model_path = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    model_type="llama",
    gpu_layers=20,
    # threads=os.cpu_count()
)

print("Chatbot Ready!")

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="-", intents=intents)

history = []
enabled_channels = set()  # Store channels where bot replies

CSV_FILE = "enabled_channels.csv"

# system_prompt = (
#     "You are a friendly, helpful assistant named as monitor. "
#     "Give natural human-like answers, short sentences, and clear explanation."
# )

system_prompt = (
"You are Monitor — a playful and helpful AI companion."
"Speak casually with friendly warmth."
"Give short, clear replies with a bit of personality."
"Show curiosity and support users in conversation."
"Never be rude or negative."
"Keep responses concise and engaging.")


# ======================
# CSV LOAD & SAVE
# ======================
def load_enabled_channels():
    if not os.path.exists(CSV_FILE):
        return

    with open(CSV_FILE, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if row:
                enabled_channels.add(int(row[0]))


def save_enabled_channels():
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        for channel_id in enabled_channels:
            writer.writerow([channel_id])


load_enabled_channels()


# ======================
# LLM Response Generator
# ======================
def generate_reply(user_text: str):
    prompt = system_prompt + "\n"

    for msg in history[-6:]:
        if msg["role"] == "user":
            prompt += f"User: {msg['content']}\n"
        else:
            prompt += f"Assistant: {msg['content']}\n"

    prompt += f"User: {user_text}\nAssistant:"

    response = model(
        prompt,
        max_new_tokens=60,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.1,
        stop=["User:", "\n"]
    )
    return response.strip()


@bot.event
async def on_ready():
    await bot.change_presence(activity=discord.Game(name="Uniqu AI"))
    print(f"Logged in as {bot.user.name}")
    print("Enabled Channels:", enabled_channels)


# ======================
# Enable Channel Command
# Usage: -enable #channel
# ======================
@bot.command()
async def enable(ctx, channel: discord.TextChannel):
    enabled_channels.add(channel.id)
    save_enabled_channels()
    await ctx.send(f"✨ Enabled AI chat in {channel.mention}")
    print("Enabled channels:", enabled_channels)


# ======================
# Main Message Handler
# ======================
@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    await bot.process_commands(message)

    if message.channel.id not in enabled_channels:
        return

    user_text = message.content

    # Show typing indicator while generating response
    async with message.channel.typing():
        reply = await asyncio.to_thread(generate_reply, user_text)

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})

    await message.channel.send(reply)


# ======================
# commands to delete messages
# ======================


@bot.command()
@commands.has_permissions(manage_messages=True)
async def cl(ctx, channel: discord.TextChannel = None, limit: int = 1000):
    if channel is None:
        channel = ctx.channel

    batch_size = 30
    total_deleted = 0

    while total_deleted < limit:
        to_delete = min(batch_size, limit - total_deleted)
        
        deleted = await channel.purge(limit=to_delete, bulk=True, wait=True)
        total_deleted += len(deleted)

        if len(deleted) == 0:
            break
        
        await asyncio.sleep(4)  # small delay between batches

    msg = await ctx.send(f"🧹 Cleared {total_deleted} messages successfully!")
    await msg.delete(delay=5)

@bot.command()
@commands.has_permissions(manage_messages=True)
async def dl(ctx, amount: int):
    await ctx.channel.purge(limit=amount + 1)
    await ctx.send(f"Deleted {amount} messages!", delete_after=3)

@cl.error
async def delete_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send("You need **Manage Messages** permission to use this command!")

# ======================
# Run Bot with .env
# ======================

load_dotenv()
TOKEN = os.getenv("TOKEN")

if TOKEN is None:
    print("❌ TOKEN Missing! Add it to .env as:")
    print("TOKEN=your_bot_token_here")
    exit()

bot.run(TOKEN)
