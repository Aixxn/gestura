import chai from 'chai';
import chaiHttp from 'chai-http';
import sinon from 'sinon';
import express from 'express';

// import the compiled router (not the TS source)
import router from '../dist/routes/api.js';

const { expect } = chai;
chai.use(chaiHttp);

describe('Express Kafka Router', function() {
  let app;
  let sendStub;

  beforeEach(function() {
    app = express();
    app.use(express.json());
    app.use('/', router);

    // stub Kafka producer.send globally
    const kafkaModule = await import('../dist/routes/api.js');
    sendStub = sinon.stub(kafkaModule.producer, 'send').resolves();
  });

  afterEach(function() {
    sinon.restore();
  });

  describe('POST /stop', function() {
    it('should return 400 if no uuid is sent', async function() {
      const res = await chai.request(app).post('/stop').send({});
      expect(res).to.have.status(400);
      expect(res.body.message).to.equal('No uuid has been sent to the server.');
    });

    it('should send to Kafka and return 200 when uuid is provided', async function() {
      sendStub.resolves();

      const res = await chai.request(app).post('/stop').send({ uuid: '1234' });
      expect(sendStub.calledOnce).to.be.true;
      expect(res).to.have.status(200);
      expect(res.text).to.equal('Successfully finished sequence.');
    });

    it('should return 500 if Kafka send fails', async function() {
      sendStub.rejects(new Error('Kafka failure'));

      const res = await chai.request(app).post('/stop').send({ uuid: '1234' });
      expect(res).to.have.status(500);
      expect(res.body.message).to.equal('Failed to stop processing.');
    });
  });

  describe('POST /convert', function() {
    it('should return 400 if no file is uploaded', async function() {
      const res = await chai.request(app).post('/convert').send({ uuid: '1234' });
      expect(res).to.have.status(400);
      expect(res.body.message).to.equal('No file has been sent to the server');
    });

    it('should return 400 if no uuid is sent', async function() {
      const res = await chai.request(app)
        .post('/convert')
        .attach('rawImage', Buffer.from('fake data'), 'fake.png');
      expect(res).to.have.status(400);
      expect(res.body.message).to.equal('No uuid sent to the server.');
    });

    it('should send image to Kafka and return 200', async function() {
      sendStub.resolves();

      const res = await chai.request(app)
        .post('/convert')
        .set('Content-Type', 'multipart/form-data')
        .attach('rawImage', Buffer.from('fake data'), 'fake.png')
        .field('uuid', '1234');

      expect(sendStub.calledOnce).to.be.true;
      expect(res).to.have.status(200);
      expect(res.body.message).to.equal('image received and queued.');
    });

    it('should return 500 if Kafka send fails', async function() {
      sendStub.rejects(new Error('Kafka error'));

      const res = await chai.request(app)
        .post('/convert')
        .set('Content-Type', 'multipart/form-data')
        .attach('rawImage', Buffer.from('fake data'), 'fake.png')
        .field('uuid', '1234');

      expect(res).to.have.status(500);
      expect(res.body.message).to.equal('Failed to proccess image. ');
    });
  });
});

