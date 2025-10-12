import express from "express";
import { Kafka } from 'kafkajs';
import multer from 'multer';

const router = express.Router();
const kafka = new Kafka({ clientId: 'rawImageProducer', brokers: ['localhost:9092'] });
const producer = kafka.producer();

const upload = multer();

(async () => {
    await producer.connect();
})();

router.get('/convert', upload.single('rawImage'), async (req, res) => {
    while (!req.body.stop) {
        await producer.send({
            topic: 'rawImageData',
            messages: [{ value: JSON.stringify(req.file) }]
        });
    }
    res.status(200);
});

