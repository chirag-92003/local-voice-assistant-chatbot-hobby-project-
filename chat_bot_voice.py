import speech_recognition as sr
import pyttsx3
import uuid
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage

# ------------------------
# Setup LLM and memory
# ------------------------
llm = ChatOllama(model="llama3:8b")
chat_history_store = {}

def chat_memory(session_id: str) -> ChatMessageHistory:
    if session_id not in chat_history_store:
        chat_history_store[session_id] = ChatMessageHistory()
    return chat_history_store[session_id]

chat = RunnableWithMessageHistory(
    llm,
    chat_memory
)

session_id = str(uuid.uuid4())

# ------------------------
# Setup TTS
# ------------------------
def speak(text: str):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print("TTS Error:", e)

# ------------------------
# Setup Google STT
# ------------------------
recognizer = sr.Recognizer()
mic = sr.Microphone()

# ------------------------
# Main loop
# ------------------------
print("Voice assistant running... Speak now. Say 'exit' to quit.")

while True:
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        continue
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        continue

    if text.lower() == "exit":
        print("Exiting...")
        break

    # Generate AI response
    try:
        response = chat.invoke(
            text,
            config={"configurable": {"session_id": session_id}}
        )
        answer = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        print("AI Error:", e)
        answer = "Sorry, I couldn't process that."

    print("AI:", answer)
    speak(answer)

# ------------------------
# Print chat history at the end
# ------------------------
print("\n--- Chat History ---")
for msg in chat_history_store[session_id].messages:
    role = "You: " if isinstance(msg, HumanMessage) else "AI: "
    print(f"{role}{msg.content}")
