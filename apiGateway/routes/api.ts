import express from "express";
import { Kafka } from 'kafkajs';
import multer from 'multer';
import { WebSocketServer, WebSocket } from 'ws'

const WEB_SOCKET_HOST = '0.0.0.0';
const WEB_SOCKET_PORT = 9898;
const KAFKA_BROKER = process.env.KAFKA_BROKER || 'kafka:9092';


const router = express.Router();
const kafka = new Kafka({ clientId: 'rawImageProducer', brokers: [ KAFKA_BROKER ] });
const producer = kafka.producer({maxInFlightRequests: 1});
const consumer = kafka.consumer({ groupId: 'translatedSignConsumer' });
const wss = new WebSocketServer({ host: WEB_SOCKET_HOST, port: WEB_SOCKET_PORT });
const sessionMap = new Map();

const upload = multer();

// kafka
(async () => {
    try {
        await producer.connect();
        console.log('Kafka producer connected successfully.')
    } catch (e) {
        console.error('Failed to connect Kafka producer:', e);
    }
})();

(async () => {
    try {
        await consumer.connect();
        await consumer.subscribe({ topic: 'translatedSign', fromBeginning: true });
        console.log('Kafka consumer connected successfully.');

        await consumer.run({
            eachMessage: async ({ message }) => {
                const resultKey = message.key ? message.key.toString() : null;
                const resultValue = message.value ? message.value.toString() : null;

                if (!resultKey) {
                    console.error('Message received does not have a Key.');
                    return;
                }
                if (!resultValue) {
                    console.error('Message received does not have a value.');
                    return;
                }

                const targetWs: WebSocket = sessionMap.get(resultKey);

                if (targetWs && targetWs.readyState == targetWs.OPEN) {
                    targetWs.send(resultValue);
                    console.log(`Pushed result for ${resultKey} to WebSocket`);
                } else{
                    console.warn(`No active socket found for result: ${resultKey}`);
                }
            }
        });
    } catch (e) {
        console.error('Failed to connect Kafka consumer.');
    }
});

// websocket server
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
router.get('/stop/:uuid', async (req, res) => {
    const uuid = req.params.uuid;

    if (!uuid) {
        console.error('No uuid sent.')
        res.status(400).send({ message: 'No uuid has been sent to the server.' })
        return
    }

    try {
        await producer.send({
            topic: 'rawImageData',
            messages: [{
                key: String(uuid),
                value: 'stop'
            }]
        })
        res.status(200).send('Successfully finished sequence.');
    } catch (e) {
        console.error('Error:', e);
        res.status(500).send({ message: 'Failed to stop processing.' });
    }
});

router.post('/convert', upload.single('rawImage'), async (req, res) => {
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
        await producer.send({
            topic: 'rawImageData',
            messages: [{
                key: String(uuid),
                value: file.buffer,
            }]
        });
        res.status(200).send({ message: 'image received and queued.' });
    } catch (e) {
        console.error('Error failed to send data to kafka broker:', e)
        res.status(500).send({ message: 'Failed to proccess image. ' })
    }
});

export default router;
