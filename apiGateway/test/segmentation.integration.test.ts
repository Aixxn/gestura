import 'mocha';
import chai from 'chai';
import sinon from 'sinon';
import supertest from 'supertest';
import axios from 'axios';
import WebSocket from 'ws';
import jwt from 'jsonwebtoken';
import app from '@src/app';

const { expect } = chai;
const request = supertest(app);

const JWT_SECRET = process.env.JWT_SECRET || 'default-dev-secret-change-in-production';

function makeToken(): string {
  return jwt.sign(
    { userId: 'test-user-id', email: 'test@example.com' },
    JWT_SECRET,
    { expiresIn: '1h' }
  );
}

describe('API Gateway ↔ Translation Service Integration', () => {
  let axiosStub: sinon.SinonStub;
  let authToken: string;

  beforeEach(() => {
    axiosStub = sinon.stub(axios, 'post');
    authToken = makeToken();
  });

  afterEach(() => {
    axiosStub.restore();
  });

  describe('POST /api/convert', () => {
    let uuid: string;
    let testCount = 0;

    beforeEach(() => {
      testCount += 1;
      uuid = `convert-test-${testCount}-${Date.now()}`;
    });

    it('returns 401 without auth token', async () => {
      const res = await request
        .post('/api/convert')
        .field('uuid', uuid)
        .expect(401);

      expect(res.body.message).to.match(/no authorization header/i);
    });

    it('returns 400 when no file (rawImage) is uploaded', async () => {
      const res = await request
        .post('/api/convert')
        .set('Authorization', `Bearer ${authToken}`)
        .field('uuid', uuid)
        .expect(400);

      expect(res.body.message).to.match(/no file/i);
    });

    it('returns 400 when no uuid is provided', async () => {
      const res = await request
        .post('/api/convert')
        .set('Authorization', `Bearer ${authToken}`)
        .attach('rawImage', Buffer.from('fake-jpeg-bytes'), 'frame.jpg')
        .expect(400);

      expect(res.body.message).to.match(/no uuid/i);
    });

    it('relays processing status from translation service', async () => {
      axiosStub.resolves({ data: { status: 'processing' } });

      const res = await request
        .post('/api/convert')
        .set('Authorization', `Bearer ${authToken}`)
        .field('uuid', uuid)
        .attach('rawImage', Buffer.from('fake-jpeg-bytes'), 'frame.jpg')
        .expect(200);

      expect(res.body).to.deep.equal({ status: 'processing' });
      expect(axiosStub.calledOnce).to.be.true;
      const callUrl = axiosStub.firstCall.args[0] as string;
      expect(callUrl).to.include('/process-frame');
    });

    it('relays word_detected status from translation service', async () => {
      axiosStub.resolves({
        data: { status: 'word_detected', word: 'hello', sign_index: 0 },
      });

      const res = await request
        .post('/api/convert')
        .set('Authorization', `Bearer ${authToken}`)
        .field('uuid', uuid)
        .attach('rawImage', Buffer.from('fake-jpeg-bytes'), 'frame.jpg')
        .expect(200);

      expect(res.body.status).to.equal('word_detected');
      expect(res.body.word).to.equal('hello');
    });

    it('encodes JPEG as base64 in the request body', async () => {
      const jpegBuffer = Buffer.from([
        0xff, 0xd8, 0xff, 0xe0,
        0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
        0xff, 0xd9,
      ]);
      axiosStub.resolves({ data: { status: 'processing' } });

      await request
        .post('/api/convert')
        .set('Authorization', `Bearer ${authToken}`)
        .field('uuid', uuid)
        .attach('rawImage', jpegBuffer, 'frame.jpg')
        .expect(200);

      const sentPayload = axiosStub.firstCall.args[1] as Record<string, unknown>;
      expect(sentPayload).to.have.property('image_bytes');
      const decoded = Buffer.from(sentPayload.image_bytes as string, 'base64');
      expect(decoded.equals(jpegBuffer)).to.be.true;
    });

    it('returns 500 when translation service is unreachable', async () => {
      axiosStub.rejects(new Error('ECONNREFUSED'));

      const res = await request
        .post('/api/convert')
        .set('Authorization', `Bearer ${authToken}`)
        .field('uuid', uuid)
        .attach('rawImage', Buffer.from('fake-jpeg-bytes'), 'frame.jpg')
        .expect(500);

      expect(res.body.message).to.match(/failed/i);
    });
  });

  describe('GET /api/stop/:uuid', () => {
    const WS_UUID = 'stop-test-ws-uuid';
    let ws: WebSocket;

    before(async function () {
      this.timeout(5000);
      const token = makeToken();
      ws = await new Promise<WebSocket>((resolve, reject) => {
        const socket = new WebSocket(`ws://localhost:9898?uuid=${WS_UUID}&token=${encodeURIComponent(token)}`);
        socket.on('open', () => resolve(socket));
        socket.on('error', reject);
        setTimeout(() => reject(new Error('WebSocket connection timeout')), 3000);
      });
    });

    after(() => {
      ws.close();
    });

    it('returns 401 without auth token', async () => {
      const res = await request
        .get('/api/stop/non-existent-session')
        .expect(401);

      expect(res.body.message).to.match(/no authorization header/i);
    });

    it('returns 404 for a UUID with no active WebSocket session', async () => {
      const res = await request
        .get('/api/stop/non-existent-session')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(404);

      expect(res.body.message).to.include('No active session');
    });

    it('returns translation result from merged service', async () => {
      const testUuid = `stop-result-${Date.now()}`;
      const token = makeToken();
      const testWs = await new Promise<WebSocket>((resolve, reject) => {
        const socket = new WebSocket(`ws://localhost:9898?uuid=${testUuid}&token=${encodeURIComponent(token)}`);
        socket.on('open', () => resolve(socket));
        socket.on('error', reject);
        setTimeout(() => reject(new Error('WS timeout')), 3000);
      });

      axiosStub.resolves({
        data: {
          asl_gloss: 'hello world',
          english: 'Hello world',
          words: ['hello', 'world'],
          success: true,
        },
      });

      const res = await request
        .get(`/api/stop/${testUuid}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(res.text).to.include('Successfully finished sequence.');

      const calledUrl = axiosStub.firstCall.args[0] as string;
      expect(calledUrl).to.include('/stop');
      const calledPayload = axiosStub.firstCall.args[1] as Record<string, unknown>;
      expect(calledPayload).to.have.property('uuid', testUuid);

      testWs.close();
    });

    it('returns 500 when translation service fails on stop', async () => {
      const testUuid = `stop-fail-${Date.now()}`;
      const token = makeToken();
      const testWs = await new Promise<WebSocket>((resolve, reject) => {
        const socket = new WebSocket(`ws://localhost:9898?uuid=${testUuid}&token=${encodeURIComponent(token)}`);
        socket.on('open', () => resolve(socket));
        socket.on('error', reject);
        setTimeout(() => reject(new Error('WS timeout')), 3000);
      });

      axiosStub.rejects(new Error('ECONNREFUSED'));

      const res = await request
        .get(`/api/stop/${testUuid}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(500);

      expect(res.body.message).to.match(/failed/i);

      testWs.close();
    });
  });
});
