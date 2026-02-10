import boto3
import json
import logging
from app.config import settings

logger = logging.getLogger('llm')


client = boto3.client("bedrock-runtime", region_name=settings.AWS_REGION)

def call_llm(prompt: str) -> str:
    logger.debug('call_llm invoked')
    logger.debug(f'Model ID: {settings.BEDROCK_MODEL_ID}')
    logger.debug(f'Prompt (first 500 chars): {prompt[:500]}')
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 800,
        "temperature": 0.1,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
    }

    response = client.invoke_model(
        modelId=settings.BEDROCK_MODEL_ID,
        body=json.dumps(body)
    )

    payload = json.loads(response["body"].read())

    logger.debug(f'Parsed payload keys: {payload.keys()}')

    try:
        text= payload['content'][0]['text']
        print("\n\n===== LLM RESPONSE =====\n", text, "\n=======================\n")
    except Exception as e:
        logger.error('Could not extract texto from payload')
        logger.error(payload)
        raise

    return text
