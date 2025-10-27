from fastapi import FastAPI

app = FastAPI(
        debug=True, 
        title='Translator Service', 
        description='This service is responsible for servince\
                the translator ai model.'
        )


@app.post('/translate')
async def translate():
    pass

@app.post('/convert-sentence')
async def convert_sentence():
    pass
