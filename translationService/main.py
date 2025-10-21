from fastapi import FastAPI

app = FastAPI(
        debug=True, 
        title='Translator Service', 
        description='This service is responsible for servince\
                the translator ai model.'
        )


@app.get('/translate')
async def translate():
    pass

@app.get('/convert-sentence')
async def convert_sentence():
    pass
