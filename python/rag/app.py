import gradio as gr
from dotenv import load_dotenv

from implementation.answer_v2 import answer_question, batch_answer_questions

load_dotenv(override=True)


def format_context(context):
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for doc in context:
        result += f"<span style='color: #ff7800;'>Source: {doc.metadata['source']}</span>\n\n"
        result += doc.page_content + "\n\n"
    return result


async def chat(history):
    last_message = str(history[-1]["content"])
    prior = history[:-1]
    
    answer, context = await answer_question(last_message, prior, tenant_id="user_test_001")
    
    history.append({"role": "assistant", "content": answer})
    return history, format_context(context)


def main():
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="Insurellm Expert Assistant") as ui:
        gr.Markdown("# 🏢 Insurellm Expert Assistant\nAsk me anything about Insurellm!")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(label="💬 Conversation", height=600)
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about Insurellm...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="📚 Retrieved Context",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )

        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot, context_markdown])

        with gr.Tab("Batch Processing"):
            gr.Markdown("### Batch Process Multiple Questions")
            batch_input = gr.Textbox(
                label="Questions (one per line)",
                placeholder="Enter questions separated by new lines...",
                lines=10
            )
            batch_output = gr.JSON(label="Results")
            batch_btn = gr.Button("Process Batch")
            
            async def process_batch(questions_str):
                questions = [q.strip() for q in questions_str.split('\n') if q.strip()]
                if not questions:
                    return {"error": "No questions provided"}
                
                results = await batch_answer_questions(questions, tenant_id="user_test_001")
                
                formatted_results = [
                    {"question": q, "answer": r[0], "has_context": len(r[1]) > 0}
                    for q, r in zip(questions, results)
                ]
                return formatted_results
            
            batch_btn.click(process_batch, inputs=batch_input, outputs=batch_output)

    ui.launch(inbrowser=True, theme=theme)

if __name__ == "__main__":
    main()