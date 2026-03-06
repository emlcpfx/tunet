const ftp = require('basic-ftp');
const path = require('path');
const fs = require('fs');

async function deploy() {
  require('dotenv').config();
  const config = {
    host: process.env.FTP_HOST,
    user: process.env.FTP_USER,
    password: process.env.FTP_PASSWORD,
    port: parseInt(process.env.FTP_PORT || '21'),
    remotePath: process.env.FTP_REMOTE_PATH
  };

  const client = new ftp.Client();
  client.ftp.verbose = true;

  try {
    await client.access({
      host: config.host,
      user: config.user,
      password: config.password,
      port: config.port,
      secure: false
    });

    const remote = config.remotePath;
    console.log(`\nConnected. Deploying to ${remote}...\n`);

    // Ensure remote directory
    await client.ensureDir(remote);

    // Upload all HTML and CSS files from this directory
    const dir = __dirname;
    const files = fs.readdirSync(dir).filter(f =>
      f.endsWith('.html') || f.endsWith('.css')
    );

    for (const file of files) {
      console.log(`  Uploading ${file}`);
      await client.uploadFrom(path.join(dir, file), remote + '/' + file);
    }

    console.log(`\nDeploy complete! ${files.length} files uploaded.`);
    console.log(`View at: https://ericmlevy.com/sparktunet/dashboard.html`);
  } catch (err) {
    console.error('Deploy failed:', err.message);
    process.exit(1);
  } finally {
    client.close();
  }
}

deploy();
