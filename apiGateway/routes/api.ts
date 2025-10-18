import express from "express";
import { Kafka } from 'kafkajs';
import multer from 'multer';

const router = express.Router();
const kafka = new Kafka({ clientId: 'rawImageProducer', brokers: ['localhost:9092'] });
const producer = kafka.producer();

const upload = multer();

(async () => {
    try {
        await producer.connect();
        console.log('Kafka producer connected successfully')
    } catch (e) {
        console.error('Failed to connect Kafka producer:', e);
    }
})();

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

