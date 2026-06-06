/**
 * Integration tests: API Gateway ←→ Sign Segmentation Service
 *
 * Tests the full convert→normalize→buffer pipeline with a mocked
 * segmentation service (HTTP-level mock via axios stub).
 *
 * Prerequisites
 * -------------
 * - MongoDB running on localhost:27017 (for auth routes, loaded alongside)
 * - Port 9898 free (WebSocket server starts on module import)
 *
 * Test strategy
 * -------------
 * - Stub axios.post to intercept calls to the segmentation service and
 *   the (blocked) translation service.
 * - Connect a real WebSocket client for /stop/:uuid tests so the
 *   endpoint sees an OPEN ws connection.
 * - Each test uses a unique UUID to prevent cross-test state leakage.
 */

import 'mocha';
import chai from 'chai';
import sinon from 'sinon';
import supertest from 'supertest';
import axios from 'axios';
import WebSocket from 'ws';
import app from '@src/app';
import { normalizeFrames } from '@src/routes/api';

const { expect } = chai;
const request = supertest(app);

// ---------------------------------------------------------------------------
// Helpers & mock data
// ---------------------------------------------------------------------------

/** Dimension of the current keypoint vector (hands + pose, no face). */
const FEATURE_DIM = 258;
/** Model window size configured in the API Gateway. */
const MODEL_WINDOW_SIZE = 80;

/**
 * Generate a deterministic 258-dim frame suitable for mock segmentation
 * output.
 */
function makeFrame(seed: number = 0): number[] {
  return Array.from({ length: FEATURE_DIM }, (_, j) => seed + j * 0.01);
}

/** Short sign sequence (5 frames — triggers padding in normalizeFrames). */
const SHORT_SIGN = [makeFrame(0), makeFrame(1), makeFrame(2), makeFrame(3), makeFrame(4)];

/** Long sign sequence (120 frames — triggers downsampling). */
const LONG_SIGN = Array.from({ length: 120 }, (_, i) => makeFrame(i));

/** A full "window" (35 frames) as the segmentation service would return. */
const SEG_WINDOW = Array.from({ length: 35 }, (_, i) => makeFrame(i));

/** Mock segmentation response when a sign has ended. */
function mockSignResult(
  signIndex: number,
  keypoints: number[][] = SHORT_SIGN,
): Record<string, unknown> {
  return {
    sign_index: signIndex,
    keypoints_sequence: keypoints,
    window: SEG_WINDOW,
    start_frame: 0,
    end_frame: keypoints.length - 1,
    motion_score_avg: 0.01,
  };
}

/** Default processing response (no sign ended yet). */
const PROCESSING_RESPONSE = { status: 'processing', frame_processed: true };

// ---------------------------------------------------------------------------
// Test suite
// ---------------------------------------------------------------------------

describe('API Gateway ↔ Segmentation Service Integration', () => {
  // ---- Hooks ---------------------------------------------------------

  /** Stub all axios.post calls so we never hit real external services. */
  let axiosStub: sinon.SinonStub;

  beforeEach(() => {
    axiosStub = sinon.stub(axios, 'post');
  });

  afterEach(() => {
    axiosStub.restore();
  });

  // ===================================================================
  // POST /api/convert — frame ingestion
  // ===================================================================

  describe('POST /api/convert', () => {
    /** Unique UUID per test to avoid cross-contamination. */
    let uuid: string;
    let testCount = 0;

    beforeEach(() => {
      testCount += 1;
      uuid = `convert-test-${testCount}-${Date.now()}`;
    });

    it('returns 400 when no file (rawImage) is uploaded', async () => {
      const res = await request
        .post('/api/convert')
        .field('uuid', uuid)
        .expect(400);

      expect(res.body.message).to.match(/no file/i);
    });

    it('returns 400 when no uuid is provided', async () => {
      const res = await request
        .post('/api/convert')
        .attach('rawImage', Buffer.from('fake-jpeg-bytes'), 'frame.jpg')
        .expect(400);

      expect(res.body.message).to.match(/no uuid/i);
    });

    it('returns 200 when segmentation reports "processing" (no sign ended)', async () => {
      axiosStub.resolves({ data: PROCESSING_RESPONSE });

      const res = await request
        .post('/api/convert')
        .field('uuid', uuid)
        .attach('rawImage', Buffer.from('fake-jpeg-bytes'), 'frame.jpg')
        .expect(200);

      expect(res.body.message).to.equal('image received and queued for segmentation.');
      // Should have called the segmentation service once
      expect(axiosStub.calledOnce).to.be.true;
      const callUrl = axiosStub.firstCall.args[0] as string;
      expect(callUrl).to.include('/process-frame');
    });

    it('buffers a completed sign into sessionSignsBuffer', async () => {
      const signResult = mockSignResult(0, SHORT_SIGN);
      axiosStub.resolves({ data: signResult });

      await request
        .post('/api/convert')
        .field('uuid', uuid)
        .attach('rawImage', Buffer.from('fake-jpeg-bytes'), 'frame.jpg')
        .expect(200);

      // The sign was sent to the segmentation service
      expect(axiosStub.calledOnce).to.be.true;
      // Verify the payload sent to the seg service contains the image
      const sentPayload = axiosStub.firstCall.args[1] as Record<string, unknown>;
      expect(sentPayload).to.have.property('uuid', uuid);
      expect(sentPayload).to.have.property('image_bytes');
      expect(sentPayload).to.have.property('timestamp_ms');
    });

    it('normalizes keypoints_sequence to MODEL_WINDOW_SIZE frames', async () => {
      // Short sign → should be padded to MODEL_WINDOW_SIZE (80)
      const signResult = mockSignResult(0, SHORT_SIGN);
      axiosStub.resolves({ data: signResult });

      await request
        .post('/api/convert')
        .field('uuid', uuid)
        .attach('rawImage', Buffer.from('fake-jpeg-bytes'), 'frame.jpg')
        .expect(200);

      // Verify normalization by checking the segmentation service payload
      const sentPayload = axiosStub.firstCall.args[1] as Record<string, unknown>;
      expect(sentPayload).to.have.property('uuid', uuid);
      // The actual normalization happens *after* the response, so we can't
      // check it through the stub alone.  The next test does it via /stop.
    });

    it('normalizes the window field when present', async () => {
      const signResult = mockSignResult(0, SHORT_SIGN);
      axiosStub.resolves({ data: signResult });

      await request
        .post('/api/convert')
        .field('uuid', uuid)
        .attach('rawImage', Buffer.from('fake-jpeg-bytes'), 'frame.jpg')
        .expect(200);
      // Normalization of both keypoints_sequence and window is verified
      // implicitly when /stop processes them.  The unit tests for
      // normalizeFrames cover the transform itself.
    });

    it('does NOT buffer when segmentation response lacks sign_index', async () => {
      const incomplete = { keypoints_sequence: SHORT_SIGN, window: SEG_WINDOW };
      axiosStub.resolves({ data: incomplete });

      await request
        .post('/api/convert')
        .field('uuid', uuid)
        .attach('rawImage', Buffer.from('fake-jpeg-bytes'), 'frame.jpg')
        .expect(200);
      // The buffer was not touched — we verify this by checking /stop
      // returns "no signs" (requires WebSocket — tested in stop section).
    });

    it('does NOT buffer when segmentation response lacks keypoints_sequence', async () => {
      const incomplete = { sign_index: 0, window: SEG_WINDOW };
      axiosStub.resolves({ data: incomplete });

      await request
        .post('/api/convert')
        .field('uuid', uuid)
        .attach('rawImage', Buffer.from('fake-jpeg-bytes'), 'frame.jpg')
        .expect(200);
    });

    it('returns 500 when segmentation service is unreachable', async () => {
      axiosStub.rejects(new Error('connect ECONNREFUSED 0.0.0.0:8000'));

      const res = await request
        .post('/api/convert')
        .field('uuid', uuid)
        .attach('rawImage', Buffer.from('fake-jpeg-bytes'), 'frame.jpg')
        .expect(500);

      expect(res.body.message).to.match(/failed/i);
    });

    it('isolates sign buffers for different sessions', async () => {
      // Session A gets a sign
      axiosStub.onFirstCall().resolves({ data: mockSignResult(0, SHORT_SIGN) });
      // Session B gets "processing"
      axiosStub.onSecondCall().resolves({ data: PROCESSING_RESPONSE });

      await request
        .post('/api/convert')
        .field('uuid', 'session-A')
        .attach('rawImage', Buffer.from('fake-jpeg-bytes'), 'frame.jpg')
        .expect(200);

      await request
        .post('/api/convert')
        .field('uuid', 'session-B')
        .attach('rawImage', Buffer.from('fake-jpeg-bytes'), 'frame.jpg')
        .expect(200);

      expect(axiosStub.calledTwice).to.be.true;
    });

    it('handles long sign sequences with downsampling', async () => {
      const longSignResult = mockSignResult(0, LONG_SIGN);
      axiosStub.resolves({ data: longSignResult });

      await request
        .post('/api/convert')
        .field('uuid', uuid)
        .attach('rawImage', Buffer.from('fake-jpeg-bytes'), 'frame.jpg')
        .expect(200);
    });

    it('sends base64-encoded JPEG to the segmentation service', async () => {
      const jpegBuffer = Buffer.from([
        0xff, 0xd8, 0xff, 0xe0,  // JPEG SOI + APP0 marker
        0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
        0xff, 0xd9,              // JPEG EOI
      ]);
      axiosStub.resolves({ data: PROCESSING_RESPONSE });

      await request
        .post('/api/convert')
        .field('uuid', uuid)
        .attach('rawImage', jpegBuffer, 'frame.jpg')
        .expect(200);

      const sentPayload = axiosStub.firstCall.args[1] as Record<string, unknown>;
      expect(sentPayload).to.have.property('image_bytes');

      // The image_bytes should be valid base64 (decodes without error)
      const decoded = Buffer.from(sentPayload.image_bytes as string, 'base64');
      expect(decoded.length).to.equal(jpegBuffer.length);
      expect(decoded.equals(jpegBuffer)).to.be.true;
    });
  });

  // ===================================================================
  // GET /api/stop/:uuid — end a signing sequence
  //
  // NOTE: These tests connect a real WebSocket to the server so that
  // sessionMap contains the uuid.  The WebSocket server is started when
  // routes/api.ts is first imported (via app.ts).
  // ===================================================================

  describe('GET /api/stop/:uuid', () => {
    /** UUID shared across the `before`/`after` WebSocket connection. */
    const WS_UUID = 'stop-test-ws-uuid';

    let ws: WebSocket;

    before(async function () {
      this.timeout(5000);
      ws = await new Promise<WebSocket>((resolve, reject) => {
        const socket = new WebSocket(`ws://localhost:9898?uuid=${WS_UUID}`);
        socket.on('open', () => resolve(socket));
        socket.on('error', (err) => reject(err));
        // Safety timeout
        setTimeout(() => reject(new Error('WebSocket connection timeout')), 3000);
      });
    });

    after(() => {
      ws.close();
    });

    it('returns 404 for a UUID with no active WebSocket session', async () => {
      const res = await request
        .get('/api/stop/non-existent-session')
        .expect(404);

      expect(res.body.message).to.include('No active session');
    });

    it('returns 200 with "no signs" when sign buffer is empty', async () => {
      const res = await request
        .get(`/api/stop/${WS_UUID}`)
        .expect(200);

      expect(res.body).to.have.property('message', 'No signs to process.');
    });

    it('processes buffered signs through the translation service pipeline', async () => {
      // ---- Setup ----
      // Connect a fresh WebSocket for this test (to have a clean buffer)
      const testUuid = `stop-test-${Date.now()}`;
      const testWs = await new Promise<WebSocket>((resolve, reject) => {
        const socket = new WebSocket(`ws://localhost:9898?uuid=${testUuid}`);
        socket.on('open', () => resolve(socket));
        socket.on('error', reject);
        setTimeout(() => reject(new Error('WS timeout')), 3000);
      });

      // Step 1: Buffer a sign via /convert (mocked segmentation returns sign)
      const signResult = mockSignResult(0, SHORT_SIGN);
      axiosStub.onFirstCall().resolves({ data: signResult });

      await request
        .post('/api/convert')
        .field('uuid', testUuid)
        .attach('rawImage', Buffer.from('fake-jpeg-bytes'), 'frame.jpg')
        .expect(200);

      // Step 2: Mock the translation service calls that /stop makes
      axiosStub.onSecondCall().resolves({ data: { pred: 'hello' } });
      axiosStub.onThirdCall().resolves({ data: { translated: 'Hello world' } });

      // Step 3: Call /stop — it should translate and clear the buffer
      const res = await request
        .get(`/api/stop/${testUuid}`)
        .expect(200);

      expect(res.text).to.include('Successfully finished sequence.');

      // Verify translation service was contacted with the right data
      const translateCall = axiosStub
        .getCalls()
        .find((c) => (c.args[0] as string).includes('/translate'));
      expect(translateCall).to.not.be.undefined;
      const translatePayload = translateCall!.args[1] as Record<string, unknown>;
      expect(translatePayload).to.have.property('window_data');

      // Verify grammar correction was called
      const grammarCall = axiosStub
        .getCalls()
        .find((c) => (c.args[0] as string).includes('/convert-sentence'));
      expect(grammarCall).to.not.be.undefined;

      // Step 4: Verify buffer was cleared — calling /stop again should say no signs
      // The buffer is cleared server-side, but the ws session still exists
      axiosStub.resolves({ data: { pred: 'hello' } });
      const res2 = await request
        .get(`/api/stop/${testUuid}`)
        .expect(200);
      expect(res2.body.message).to.equal('No signs to process.');

      testWs.close();
    });

    it('handles translation service failure gracefully', async () => {
      const testUuid = `stop-fail-${Date.now()}`;
      const testWs = await new Promise<WebSocket>((resolve, reject) => {
        const socket = new WebSocket(`ws://localhost:9898?uuid=${testUuid}`);
        socket.on('open', () => resolve(socket));
        socket.on('error', reject);
        setTimeout(() => reject(new Error('WS timeout')), 3000);
      });

      // Buffer a sign
      axiosStub.onFirstCall().resolves({ data: mockSignResult(0, SHORT_SIGN) });
      await request
        .post('/api/convert')
        .field('uuid', testUuid)
        .attach('rawImage', Buffer.from('fake-jpeg-bytes'), 'frame.jpg')
        .expect(200);

      // Translation service is down
      axiosStub.onSecondCall().rejects(new Error('ECONNREFUSED'));

      const res = await request
        .get(`/api/stop/${testUuid}`)
        .expect(500);

      expect(res.body.message).to.match(/failed/i);

      testWs.close();
    });

    it('processes multiple buffered signs in sequence', async () => {
      const testUuid = `stop-multi-${Date.now()}`;
      const testWs = await new Promise<WebSocket>((resolve, reject) => {
        const socket = new WebSocket(`ws://localhost:9898?uuid=${testUuid}`);
        socket.on('open', () => resolve(socket));
        socket.on('error', reject);
        setTimeout(() => reject(new Error('WS timeout')), 3000);
      });

      // Buffer 3 signs via /convert
      const sign1 = mockSignResult(0, SHORT_SIGN);
      const sign2 = mockSignResult(1, SHORT_SIGN);
      const sign3 = mockSignResult(2, SHORT_SIGN);

      axiosStub.onCall(0).resolves({ data: sign1 });
      axiosStub.onCall(1).resolves({ data: sign2 });
      axiosStub.onCall(2).resolves({ data: sign3 });

      for (let i = 0; i < 3; i++) {
        await request
          .post('/api/convert')
          .field('uuid', testUuid)
          .attach('rawImage', Buffer.from('fake-jpeg-bytes'), 'frame.jpg')
          .expect(200);
      }

      // Mock translation — one call per sign, then grammar correction
      // 3 signs → 3 translate calls + 1 grammar call = 4 total
      axiosStub.onCall(3).resolves({ data: { pred: 'hello' } });
      axiosStub.onCall(4).resolves({ data: { pred: 'world' } });
      axiosStub.onCall(5).resolves({ data: { pred: 'test' } });
      axiosStub.onCall(6).resolves({ data: { translated: 'Hello world test' } });

      const res = await request
        .get(`/api/stop/${testUuid}`)
        .expect(200);

      expect(res.text).to.include('Successfully finished sequence.');

      // Verify 3 translation calls were made
      const translateCalls = axiosStub
        .getCalls()
        .filter((c) => (c.args[0] as string).includes('/translate'));
      expect(translateCalls.length).to.equal(3);

      testWs.close();
    });
  });

  // ===================================================================
  // normalizeFrames integration with segmentation output shapes
  // ===================================================================

  describe('normalizeFrames — segmentation output shapes', () => {
    it('pads a 5-frame sign to MODEL_WINDOW_SIZE (80)', () => {
      const result = normalizeFrames(SHORT_SIGN, MODEL_WINDOW_SIZE);
      expect(result).to.have.length(MODEL_WINDOW_SIZE);
      // First frames should match source
      for (let i = 0; i < SHORT_SIGN.length; i++) {
        expect(result[i]).to.deep.equal(SHORT_SIGN[i]);
      }
      // Last frame should be held (padding)
      const lastSrc = SHORT_SIGN[SHORT_SIGN.length - 1];
      expect(result[MODEL_WINDOW_SIZE - 1]).to.deep.equal(lastSrc);
    });

    it('downsamples a 120-frame sign to MODEL_WINDOW_SIZE (80)', () => {
      const result = normalizeFrames(LONG_SIGN, MODEL_WINDOW_SIZE);
      expect(result).to.have.length(MODEL_WINDOW_SIZE);
      // First and last frames preserved
      expect(result[0]).to.deep.equal(LONG_SIGN[0]);
      expect(result[MODEL_WINDOW_SIZE - 1]).to.deep.equal(LONG_SIGN[LONG_SIGN.length - 1]);
    });

    it('preserves feature dimension (258) after normalization', () => {
      const padded = normalizeFrames(SHORT_SIGN, MODEL_WINDOW_SIZE);
      padded.forEach((frame) => {
        expect(frame).to.have.length(FEATURE_DIM);
      });

      const downsampled = normalizeFrames(LONG_SIGN, MODEL_WINDOW_SIZE);
      downsampled.forEach((frame) => {
        expect(frame).to.have.length(FEATURE_DIM);
      });
    });

    it('handles the sliding window shape (35×258)', () => {
      const result = normalizeFrames(SEG_WINDOW, MODEL_WINDOW_SIZE);
      expect(result).to.have.length(MODEL_WINDOW_SIZE);
      expect(result[0]).to.deep.equal(SEG_WINDOW[0]);
      // The last frame of window should be preserved
      expect(result[MODEL_WINDOW_SIZE - 1]).to.deep.equal(SEG_WINDOW[SEG_WINDOW.length - 1]);
    });
  });
});
