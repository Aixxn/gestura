import express from "express";
import { Kafka } from 'kafkajs';
import multer from 'multer';
import { WebSocketServer } from 'ws'
import redis from 'redis'

const router = express.Router();
const kafka = new Kafka({ clientId: 'rawImageProducer', brokers: ['localhost:9092'] });
const producer = kafka.producer();
const consumer = kafka.consumer({ groupId: 'translatedSignConsumer' });
const wss = new WebSocketServer({ port: 9898 });
const redisClient = redis.createClient({ url: 'redis://localhost:6379' });

const upload = multer();

// redis
redisClient.on('connect', () => console.log('Connected to redis'));
redisClient.on('error', (e) => console.error('Redis client error:', e));

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
    try{
        await consumer.connect();
        await consumer.subscribe({ topic: 'translatedSign', fromBeginning: true })
        console.log('Kafka consumer connected successfully.');
    } catch(e){
        console.error('Failed to connect Kafka consumer.');
    }
});

// TODO: continue here
wss.on('connection', async (ws) => {
    ws.on('message', (data) => {
        console.log('Message received:', data);
    });
    await consumer.run({
        eachMessage: async ({ message }) => {
            ws.send(String(message));  //still not sure of the data that will be received
        },
    })
});

router.post('/stop', async (req, res) => {
    const uuid = req.body.uuid;

    if (!uuid){
        console.error('No uuid sent.');
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

