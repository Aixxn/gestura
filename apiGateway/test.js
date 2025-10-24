const chai = require('chai');
const chaiHttp = require('chai-http');
const app = require('./dist/app.js'); // Adjust if your app file path differs

chai.use(chaiHttp);
const { expect } = chai;

describe('API Routes', () => {
  describe('POST /stop', () => {
    it('should return an error when no UUID is provided', async () => {
      const res = await chai.request(app).post('/stop').send({});
      expect(res).to.have.status(400);
      expect(res.body).to.have.property('error');
    });

    it('should return success when UUID is provided', async () => {
      const res = await chai.request(app).post('/stop').send({ uuid: '1234' });
      expect(res).to.have.status(200);
      expect(res.body).to.have.property('message');
    });
  });

  // ----------- /convert route tests -----------
  describe('POST /convert', function () {
    it('should return 400 if no file is uploaded', async function () {
      const res = await chai.request(app).post('/convert').send({ uuid: '1234' });
      expect(res).to.have.status(400);
      expect(res.body.message).to.equal('No file has been sent to the server');
    });

    it('should return 400 if no uuid is sent', async function () {
      const res = await chai.request(app)
        .post('/convert')
        .attach('rawImage', Buffer.from('fake data'), 'fake.png');
      expect(res).to.have.status(400);
      expect(res.body.message).to.equal('No uuid sent to the server.');
    });

    it('should send image to Kafka and return 200', async function () {
      const res = await chai.request(app)
        .post('/convert')
        .set('Content-Type', 'multipart/form-data')
        .attach('rawImage', Buffer.from('fake data'), 'fake.png')
        .field('uuid', '1234');
      expect(res).to.have.status(200);
      expect(res.body.message).to.equal('image received and queued.');
    });

    it('should return 500 if Kafka send fails', async function () {
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

