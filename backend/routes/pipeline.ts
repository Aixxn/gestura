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

router.post('/convert/', upload.single('rawImage'), async (req, res) => {
    while (!req.body.stop) {
        const file = req.file 

        if (!file){
            console.error('No data uploaded');
            res.status(400);
            return
        }

        await producer.send({
            topic: 'rawImageData',
            messages: [{
                key: JSON.stringify(req.body.uuid),
                value: JSON.stringify(file?.buffer),
            }]
        });
    } 
    res.status(200);
});

