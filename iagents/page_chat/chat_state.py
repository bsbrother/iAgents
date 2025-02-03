from __future__ import annotations

import datetime
import functools
import os

import pytz
import reflex as rx
from openai import OpenAI
from sqlalchemy import or_, select
from together import Together

from .chat_messages.model_chat_interaction import ChatInteraction

from dotenv import load_dotenv

load_dotenv()

AI_MODEL: str = "UNKNOWN"

@functools.lru_cache
def get_ai_client() -> OpenAI | Together:
    ai_provider = os.environ.get("AI_PROVIDER")
    match ai_provider:
        case "deepseek":
            return OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
            )

        case "openai":
            return OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
            )

        case "together":
            return Together(
                api_key=os.environ.get("TOGETHER_API_KEY"),
            )

        case _:
            print("Invalid AI provider. Please set AI_PROVIDER environment variable")


def get_ai_chat_completion_kwargs() -> dict:
    ai_provider = os.environ.get("AI_PROVIDER")
    common = dict(
        max_tokens=512,
        temperature=0.7,
        top_p=0.7,
        stop=["<|eot_id|>", "<|eom_id|>"],
        stream=True,
    )
    if ai_provider == "together":
        common.update(
            dict(
                top_k=50,
                repetition_penalty=1,
                truncate=130560,
            )
        )
    return common


def get_ai_model() -> None:
    global AI_MODEL
    ai_model = os.environ.get("AI_PROVIDER")
    match ai_model:
        case "deepseek":
            AI_MODEL = "deepseek-reasoner"

        case "openai":
            AI_MODEL = "gpt-3.5-turbo"

        case "together":
            AI_MODEL = "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"

        case _:
            print("Invalid AI provider. Please set AI_PROVIDER environment variable")


get_ai_model()

MAX_QUESTIONS = 10
INPUT_BOX_ID = "input-box"


class ChatState(rx.State):
    """The app state."""

    filter: str = ""

    _ai_chat_instance = None

    chat_interactions: list[ChatInteraction] = []

    has_token: bool = True
    username: str = "Mauro Sicard"
    prompt: str = ""
    result: str = ""
    ai_loading: bool = False
    timestamp: datetime.datetime = datetime.datetime.now(
        tz=pytz.timezone(
            "US/Pacific",
        ),
    )

    @rx.var
    def timestamp_formatted(
        self,
    ) -> str:
        return self.timestamp.strftime("%I:%M %p")

    def _get_client_instance(
        self,
    ) -> OpenAI | Together:
        if ai_client_instance := get_ai_client():
            return ai_client_instance

        raise ValueError("AI client not found")

    def _fetch_messages(
        self,
    ) -> list[ChatInteraction]:
        with rx.session() as session:
            query = select(ChatInteraction)
            if self.filter:
                query = query.where(
                    or_(
                        ChatInteraction.prompt.ilike(f"%{self.filter}%"),
                        ChatInteraction.answer.ilike(f"%{self.filter}%"),
                    ),
                )

            return (
                session.exec(
                    query.distinct(ChatInteraction.prompt)
                    .order_by(ChatInteraction.timestamp.asc())
                    .limit(MAX_QUESTIONS),
                )
                .scalars()
                .all()
            )

    def load_messages_from_database(
        self,
    ) -> None:
        self.chat_interactions = self._fetch_messages()

    def set_prompt(
        self,
        prompt: str,
    ) -> None:
        self.prompt = prompt

    def create_new_chat(
        self,
    ) -> None:
        pass

    def _save_resulting_chat_interaction(
        self,
        chat_interaction: ChatInteraction,
    ) -> None:
        with rx.session() as session:
            session.add(
                chat_interaction,
            )
            session.commit()
            session.refresh(chat_interaction)

    async def _check_saved_chat_interactions(
        self,
        username: str,
        prompt: str,
    ) -> None:
        with rx.session() as session:
            if (
                session.exec(
                    select(ChatInteraction)
                    .where(
                        ChatInteraction.chat_participant_user_name == username,
                    )
                    .where(
                        ChatInteraction.prompt == prompt,
                    ),
                ).first()
                or len(
                    session.exec(
                        select(ChatInteraction)
                        .where(
                            ChatInteraction.chat_participant_user_name == username,
                        )
                        .where(
                            ChatInteraction.timestamp
                            > datetime.datetime.now()
                            - datetime.timedelta(
                                days=1,
                            ),
                        ),
                    ).all(),
                )
                > MAX_QUESTIONS
            ):
                raise ValueError(
                    "You have already asked this question or have asked too many questions in the past 24 hours.",
                )

    @rx.event(background=True)
    async def submit_prompt(
        self,
    ):
        async def _fetch_chat_completion_session(
            prompt: str,
        ):
            def _create_messages_for_chat_completion():
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are a helpful assistant. Respond in markdown.",
                            },
                        ],
                    },
                ]
                for chat_interaction in self.chat_interactions:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": chat_interaction.prompt,
                                },
                            ],
                        },
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": chat_interaction.answer,
                                },
                            ],
                        },
                    )

                messages.append(
                    {
                        "role": "user",
                        "content": prompt,
                    },
                )
                return messages

            messages = _create_messages_for_chat_completion()
            ai_client_instance = self._get_client_instance()
            stream = ai_client_instance.chat.completions.create(
                model=AI_MODEL,
                messages=messages,
                **get_ai_chat_completion_kwargs(),
            )
            if stream is None:
                raise Exception("Session is None")

            return stream

        def set_ui_loading_state() -> None:
            self.ai_loading = True

        def clear_ui_loading_state() -> None:
            self.result = ""
            self.ai_loading = False

        def add_new_chat_interaction() -> None:
            self.chat_interactions.append(
                ChatInteraction(
                    prompt=prompt,
                    answer="",
                    chat_participant_user_name=self.username,
                ),
            )
            self.prompt = ""

        # Get the question from the form
        if self.prompt == "":
            return

        prompt = self.prompt
        if self.username == "":
            raise ValueError("Username is required")

        await self._check_saved_chat_interactions(
            prompt=prompt,
            username=self.username,
        )
        async with self:
            set_ui_loading_state()

            yield

            stream = await _fetch_chat_completion_session(prompt)
            clear_ui_loading_state()
            add_new_chat_interaction()
            yield

            try:
                for item in stream:
                    if item.choices and item.choices[0] and item.choices[0].delta:
                        answer_text = item.choices[0].delta.content
                        # Ensure answer_text is not None before concatenation
                        if answer_text is not None:
                            self.chat_interactions[-1].answer += answer_text

                        else:
                            answer_text = ""
                            self.chat_interactions[-1].answer += answer_text

                        yield rx.scroll_to(
                            elem_id=INPUT_BOX_ID,
                        )

            except StopAsyncIteration:
                raise

            self.result = self.chat_interactions[-1].answer

        self._save_resulting_chat_interaction(
            chat_interaction=self.chat_interactions[-1],
        )
