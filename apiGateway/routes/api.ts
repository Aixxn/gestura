import express from "express";
import multer from 'multer';
import { WebSocketServer } from 'ws'
import axios from 'axios';

const WEB_SOCKET_HOST = process.env.WEB_SOCKET_PORT || '0.0.0.0';
const WEB_SOCKET_PORT = parseInt(process.env.WEB_SOCKET_HOST || '9898');
const SEGMENTATION_SERVICE_URL = process.env.SEGMENTATION_SERVICE_URL || 'http://signsegmentationservice:8000';
const MODEL_WINDOW_SIZE = parseInt(process.env.MODEL_WINDOW_SIZE || '80');
const MODEL_FEATURE_DIM = parseInt(process.env.MODEL_FEATURE_DIM || '1663');

const translationRouter = express.Router();
const wss = new WebSocketServer({ host: WEB_SOCKET_HOST, port: WEB_SOCKET_PORT });
const sessionMap = new Map(); // Maps UUID to WebSocket
const sessionSignsBuffer = new Map(); // Maps UUID to array of signs for current sentence

/**
 * Normalize a variable-length sign segment to the model's expected window size.
 *
 * - If too long: uniformly downsample (preserves temporal profile)
 * - If too short: repeat the last frame (natural held-position padding)
 * - If exact match: return as-is
 */
export function normalizeFrames(frames: number[][], targetFrames: number, _featureDim: number): number[][] {
  const n = frames.length;
  if (n === 0) return [];
  if (n === targetFrames) return frames;
  if (targetFrames === 1) return [frames[0]!];

  if (n > targetFrames) {
    const indices = Array.from({ length: targetFrames }, (_, i) =>
      Math.round((n - 1) * i / (targetFrames - 1))
    );
    return indices.map(i => frames[i]!);
  }

  // Pad by repeating the last frame (natural "held sign" behavior)
  const padded = frames.slice();
  const last = frames[frames.length - 1]!;
  while (padded.length < targetFrames) {
    padded.push([...last]);
  }
  return padded;
}

const upload = multer();

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
        sessionSignsBuffer.delete(clientUuid);
        console.log(`Client ${clientUuid} has disconnected.`);
    });

    sessionMap.set(clientUuid, ws);
    sessionSignsBuffer.set(clientUuid, []); // Initialize empty signs buffer for this session
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
        // Get buffered signs for this session
        const signs = sessionSignsBuffer.get(uuid) || [];
        
        if (signs.length === 0) {
            res.status(200).send({ message: 'No signs to process.' });
            return;
        }

        // Translate each sign through the ML model
        const predictedWords: string[] = [];
        for (const sign of signs) {
            const translateResponse = await axios.post(
                'http://translationService:7860/translate',
                { window_data: sign.keypoints_sequence },
                { timeout: 10000 }
            );
            if (translateResponse.data && translateResponse.data.pred) {
                predictedWords.push(translateResponse.data.pred);
            }
        }

        // Convert ASL gloss to natural English
        const aslGloss = predictedWords.join(' ');
        const grammarResponse = await axios.post(
            'http://translationService:7860/convert-sentence',
            { asl_gloss: aslGloss },
            { timeout: 10000 }
        );

        const result = {
            type: 'translation',
            asl_gloss: aslGloss,
            english: grammarResponse.data?.translated || aslGloss,
            words: predictedWords,
        };

        // Send result to client
        if (ws.readyState === ws.OPEN) {
            ws.send(JSON.stringify(result));
        }

        // Clear buffer for this session
        sessionSignsBuffer.set(uuid, []);
        
        res.status(200).send('Successfully finished sequence.');
    } catch (e) {
        console.error('Error:', e);
        res.status(500).send({ message: 'Failed to proccess image. ' });
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
        // Convert image buffer to base64 for sending to segmentation service
        const imageBase64 = file.buffer.toString('base64');
        
        // Send frame to segmentation service
        const segmentationResponse = await axios.post(
            `${SEGMENTATION_SERVICE_URL}/process-frame`,
            {
                uuid: uuid,
                image_bytes: imageBase64,
                timestamp_ms: Date.now()
            },
            { timeout: 5000 }
        );

        // If we got sign data back, normalize and buffer it for this session
        if (segmentationResponse.data && 
            segmentationResponse.data.sign_index !== undefined &&
            segmentationResponse.data.keypoints_sequence) {
            
            const rawSign = segmentationResponse.data;
            
            // Normalize variable-length keypoint sequence to model's expected window size
            const normalizedSign = {
                ...rawSign,
                keypoints_sequence: normalizeFrames(
                    rawSign.keypoints_sequence,
                    MODEL_WINDOW_SIZE,
                    MODEL_FEATURE_DIM
                ),
                // Also normalize the window snapshot if present
                window: rawSign.window
                    ? normalizeFrames(rawSign.window, MODEL_WINDOW_SIZE, MODEL_FEATURE_DIM)
                    : rawSign.window,
            };
            
            const signsBuffer = sessionSignsBuffer.get(uuid) || [];
            signsBuffer.push(normalizedSign);
            sessionSignsBuffer.set(uuid, signsBuffer);
        }

        res.status(200).send({ message: 'image received and queued for segmentation.' });
    } catch (e) {
        console.error('Error failed to send data to segmentation broker:', e)
        res.status(500).send({ message: 'Failed to process image. ' })
    }
});

export default translationRouter;