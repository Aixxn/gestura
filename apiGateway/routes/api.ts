import express from 'express';
import multer from 'multer';
import { WebSocketServer } from 'ws'
import axios from 'axios';
import jwt from 'jsonwebtoken';
import { requireAuth } from 'middleware/auth';
import { AuthRequest, JwtPayload } from 'types/index';
import logger from 'services/logger';
import { withLogging, setupWsLogging } from 'middleware/logging';

const WEB_SOCKET_HOST = process.env.WEB_SOCKET_HOST || '0.0.0.0';
const WEB_SOCKET_PORT = parseInt(process.env.WEB_SOCKET_PORT || '9898');
const TRANSLATION_SERVICE_URL = process.env.TRANSLATION_SERVICE_URL || 'http://translationService:7860';
const JWT_SECRET = process.env.JWT_SECRET || 'default-dev-secret-change-in-production';

const translationRouter = express.Router();
const wss = new WebSocketServer({ host: WEB_SOCKET_HOST, port: WEB_SOCKET_PORT });
const sessionMap = new Map(); // Maps UUID to WebSocket

const upload = multer();

// websocket server — tracks active sessions for stop endpoint
wss.on('connection', async (ws, req) => {
  const urlParams = new URLSearchParams(req.url?.split('?')[1]);
  const clientUuid = urlParams.get('uuid');
  const token = urlParams.get('token');

  if (!clientUuid) {
    ws.close(1008, 'UUID is required for this connection.');
    return;
  }

  // Validate JWT for WebSocket connections
  if (!token) {
    ws.close(1008, 'Authentication token is required.');
    return;
  }

  try {
    jwt.verify(token, JWT_SECRET) as JwtPayload;
  } catch {
    ws.close(1008, 'Invalid or expired authentication token.');
    return;
  }

  ws.on('close', async () => {
    sessionMap.delete(clientUuid);
  });

  sessionMap.set(clientUuid, ws);
});

setupWsLogging(wss);

// endpoints
translationRouter.get('/stop/:uuid', requireAuth, withLogging(async (req: AuthRequest, res) => {
  const gatewayStopReceivedAt = Date.now();
  const uuid = req.params.uuid;
  const ws = sessionMap.get(uuid);

  if (!uuid) {
    logger.warn('No uuid sent.')
    res.status(400).send({ message: 'No uuid has been sent to the server.' })
    return
  }

  if (!ws) {
    res.status(404).send({ message: 'No active session for this UUID.' });
    return;
  }

  try {
    const translationStopStartedAt = Date.now();
    const response = await axios.post(
      `${TRANSLATION_SERVICE_URL}/stop`,
      { uuid },
      { timeout: 15000 }
    );
    const translationStopFinishedAt = Date.now();
    const responseData: Record<string, any> = response.data && typeof response.data === 'object'
      ? response.data
      : {};
    const timing = {
      ...(responseData.timing || {}),
      gateway_stop_received_at_ms: gatewayStopReceivedAt,
      gateway_stop_translation_roundtrip_ms: translationStopFinishedAt - translationStopStartedAt,
      gateway_stop_total_ms: Date.now() - gatewayStopReceivedAt,
      gateway_ws_send_at_ms: Date.now(),
    };

    const result = {
      type: 'translation',
      asl_gloss: responseData.asl_gloss || '',
      english: responseData.english || '',
      words: responseData.words || [],
      timing,
    };

    if (ws.readyState === ws.OPEN) {
      ws.send(JSON.stringify(result));
    }

    logger.info('[LATENCY] /api/stop timing:', {
      uuid,
      ...timing,
    });

    res.status(200).send('Successfully finished sequence.');
  } catch (e) {
    logger.error('Failed to process sequence', { error: e });
    res.status(500).send({ message: 'Failed to process sequence.' })
  }
}, 'api.stop'));

translationRouter.post('/convert', requireAuth, upload.single('rawImage'), withLogging(async (req: AuthRequest, res) => {
  const gatewayReceivedAt = Date.now();
  const file = req.file
  const uuid = req.body.uuid

  if (!file) {
    logger.warn('No data uploaded');
    res.status(400).send({ message: 'No file has been sent to the server' });
    return
  }
  if (!uuid) {
    logger.warn('No uuid sent');
    res.status(400).send({ message: 'No uuid sent to the server.' });
    return;
  }

  try {
    const imageBase64 = file.buffer.toString('base64');
    const translationRequestStartedAt = Date.now();

    const response = await axios.post(
      `${TRANSLATION_SERVICE_URL}/process-frame`,
      {
        uuid: uuid,
        image_bytes: imageBase64,
        timestamp_ms: gatewayReceivedAt
      },
      { timeout: 5000 }
    );
    const translationResponseReceivedAt = Date.now();
    const responseData: Record<string, any> = response.data && typeof response.data === 'object'
      ? response.data
      : {};
    const timing = {
      ...(responseData.timing || {}),
      gateway_received_at_ms: gatewayReceivedAt,
      gateway_translation_request_started_at_ms: translationRequestStartedAt,
      gateway_translation_response_received_at_ms: translationResponseReceivedAt,
      gateway_translation_roundtrip_ms: translationResponseReceivedAt - translationRequestStartedAt,
      gateway_total_ms: Date.now() - gatewayReceivedAt,
      upload_bytes: file.size,
    };

    logger.info('[LATENCY] /api/convert timing:', {
      uuid,
      status: responseData.status,
      ...timing,
    });

    res.status(200).send({
      ...responseData,
      timing,
    });
  } catch (e) {
    logger.error('Failed to send data to translation service', { error: e })
    res.status(500).send({ message: 'Failed to process image. ' })
  }
}, 'api.convert'));

export default translationRouter;
