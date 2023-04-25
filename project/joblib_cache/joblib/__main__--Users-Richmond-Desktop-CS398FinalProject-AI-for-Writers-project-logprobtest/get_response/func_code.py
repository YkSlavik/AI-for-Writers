# first line: 16
@memory.cache
def get_response(full_request):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=full_request,
        temperature=0.7,
        max_tokens=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=10,
        echo=True
    )
    return response
