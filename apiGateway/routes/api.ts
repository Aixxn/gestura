import express from "express";
import multer from 'multer';
import { WebSocketServer } from 'ws'
import axios from 'axios';

const WEB_SOCKET_HOST = process.env.WEB_SOCKET_HOST || '0.0.0.0';
const WEB_SOCKET_PORT = parseInt(process.env.WEB_SOCKET_PORT || '9898');
const TRANSLATION_SERVICE_URL = process.env.TRANSLATION_SERVICE_URL || 'http://translationService:7860';

const translationRouter = express.Router();
const wss = new WebSocketServer({ host: WEB_SOCKET_HOST, port: WEB_SOCKET_PORT });
const sessionMap = new Map(); // Maps UUID to WebSocket

const upload = multer();

// websocket server — tracks active sessions for stop endpoint
wss.on('connection', async (ws, req) => {
    const urlParams = new URLSearchParams(req.url?.split('?')[1]);
    const clientUuid = urlParams.get('uuid');

    if (!clientUuid) {
        ws.close(1008, 'UUID is required for this connection.');
        return;
    }

    ws.on('close', async () => {
        sessionMap.delete(clientUuid);
        console.log(`Client ${clientUuid} has disconnected.`);
    });

    sessionMap.set(clientUuid, ws);
    console.log(`Client ${clientUuid} has connected.`);
});

// endpoints
translationRouter.get('/stop/:uuid', async (req, res) => {
    const uuid = req.params.uuid;
    const ws = sessionMap.get(uuid);

    if (!uuid) {
        console.error('No uuid sent.')
        res.status(400).send({ message: 'No uuid has been sent to the server.' })
        return
    }

    if (!ws) {
        res.status(404).send({ message: 'No active session for this UUID.' });
        return;
    }

    try {
        const response = await axios.post(
            `${TRANSLATION_SERVICE_URL}/stop`,
            { uuid },
            { timeout: 15000 }
        );

        const result = {
            type: 'translation',
            asl_gloss: response.data.asl_gloss || '',
            english: response.data.english || '',
            words: response.data.words || [],
        };

        if (ws.readyState === ws.OPEN) {
            ws.send(JSON.stringify(result));
        }

        res.status(200).send('Successfully finished sequence.');
    } catch (e) {
        console.error('Error:', e);
        res.status(500).send({ message: 'Failed to process sequence.' })
    }
});

translationRouter.post('/convert', upload.single('rawImage'), async (req, res) => {
    const file = req.file
    const uuid = req.body.uuid

    if (!file) {
        console.error('No data uploaded');
        res.status(400).send({ message: 'No file has been sent to the server' });
        return
    }
    if (!uuid) {
        console.error('No uuid sent');
        res.status(400).send({ message: 'No uuid sent to the server.' });
        return;
    }

    try {
        const imageBase64 = file.buffer.toString('base64');

        const response = await axios.post(
            `${TRANSLATION_SERVICE_URL}/process-frame`,
            {
                uuid: uuid,
                image_bytes: imageBase64,
                timestamp_ms: Date.now()
            },
            { timeout: 5000 }
        );

        res.status(200).send(response.data);
    } catch (e) {
        console.error('Error failed to send data to translation service:', e)
        res.status(500).send({ message: 'Failed to process image. ' })
    }
});

export default translationRouter;
