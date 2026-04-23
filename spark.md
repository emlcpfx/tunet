Spark Cloud Studio - API Documentation
This documentation provides a comprehensive guide to the Spark Cloud Studio API for creating and
managing Ubuntu 24 workstations. Follow the examples below to integrate with our API.

Table of Contents
Authentication  Getting an Access Token
Creating a New Ubuntu 24 Workstation
Deploying Your Ubuntu 24 Workstation
Listing Workstations
Starting and Stopping a Workstation
Deleting a Workstation
1. Authentication - Getting an Access Token
Before using the Spark Cloud Studio API, you need to obtain an access token for authentication.

Endpoint
POST https://api.prod.aapse1.sparkcloud.studio/auth/login
Request Headers
{
"Content-Type": "application/json"
}
Request Body
{
"email": "string",
"password": "string"
}
Response
The response will contain an access token JWT in the JSON body. Copy the token value and use it in the
Authorization header for all subsequent API calls.

{
"resp": "Login Successful",
"success": true,
"access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
"password_expired": false,
"password_expires_in_days": null,
"requires_password_change": false
}
Example (cURL)
curl  X POST https://api.prod.aapse1.sparkcloud.studio/auth/login \
H "accept: application/json" \
H "Content-Type: application/json" \
-d '{
"email": "YOUR_EMAIL_ADDRESS",
"password": "YOUR_ACCOUNT_PASSWORD"
}'
Note: Store the access token securely and include it in the Authorization header for all subsequent API
requests.

2. Creating a New Ubuntu 24 Workstation
Create a new Ubuntu 24 workstation for your account. This creates a temporary workstation that must be
deployed in the next step.

Endpoint
POST https://api.prod.aapse1.sparkcloud.studio/api/supabase/workstation/store-temp-workstation?conte
xt=app
Request Headers
{
"Content-Type": "application/json",
"Authorization": "Bearer YOUR_ACCESS_TOKEN"
}
Request Body
{
"region": "ap-southeast-1", // Deployment region code
"workstation_static_id" 163, // ID for config preset
"os_type": "ubuntu24",
"app_type": "base",
"template_version": null
}
Parameters
region (string, optional): The deployment region where the workstation will be created. Must be from
these supported options:
Region Code Location
ap-south-1 Mumbai
ap-northeast-2 Seoul
ap-southeast-3 Jakarta
ap-northeast-1 Tokyo
sa-east-1 São Paulo
ca-central-1 Montreal
ap-southeast-1 Singapore
eu-west-2 London
us-west-2 Oregon
us-west-1 California
af-south-1 Cape Town
eu-central-1 Frankfurt
ap-southeast-2 Sydney
me-central-1 Dubai
me-south-1 Bahrain
us-east-1 Virginia
workstation_static_id (integer, required): An ID for a specific workstation configuration preset. Must match one
of the following available options.
Workstation Static ID Name GPU CPU RAM
173 Exceptional power 1 4x NVIDIA T4 2560 CUD4 16 GBA cores Intel Xeon Platinum 348 cores .5GHz 192 GB
171 Top-tier 4 NVIDIA T2560 CUD4 16 GBA cores Intel Xeon Platinum 364 cores .5GHz 256 GB
165 Mid-tier 1 NVIDIA T2560 CUD4 16 GBA cores Intel Xeon Platinum 38 cores .5GHz 32 GB
167 High-end studio 1 NVIDIA T2560 CUD4 16 GBA cores Intel Xeon Platinum 316 cores .5GHz 64 GB
169 Top-tier 1 NVIDIA T2560 CUD4 16 GBA cores Intel Xeon Platinum 332 cores .5GHz 128 GB
172 Top-tier 5 NVIDIA A10 24 GB
9216 CUDA cores
AMD Zen2 EPYC 3.3GHz
64 cores
256 GB
176 Exceptional power 4 8x NVIDIA T8 2560 CUD4 16 GBA cores Intel Xeon Platinum 396 cores .5GHz 384 GB
164 Starter kit 2 NVIDIA A9216 CUD10 24 GBA cores AMD Zen2 EPY4 cores C 3.3GHz 16 GB
163 Starter kit 1
NVIDIA T4 16 GB
2560 CUDA cores
Intel Xeon Platinum 3.5GHz
4 cores 16 GB
175 Exceptional power 3
4x NVIDIA A10 24 GB
4 9216 CUDA cores
AMD Zen2 EPYC 3.3GHz
96 cores 384 GB
166 Mid-tier 3 NVIDIA A9216 CUD10 24 GBA cores AMD Zen2 EPY8 cores C 3.3GHz 32 GB
206 Top-tier 6 NVIDIA L40S 48 GB18176 CUDA cores AMD Zen3 EPY64 cores C 3.7GHz 512 GB
204 High-end studio 3 NVIDIA L40S 48 GB18176 CUDA cores AMD Zen3 EPY16 cores C 3.7GHz 128 GB
205 Top-tier 3 NVIDIA L40S 48 GB18176 CUDA cores AMD Zen3 EPY32 cores C 3.7GHz 256 GB
174 Exceptional power 2 4x NVIDIA A4 9216 CUD10 24 GBA cores AMD Zen2 EPY48 cores C 3.3GHz 192 GB
208 High memorrendering y NVIDIA L4 27424 CUDA cores4 GB AMD Zen3 EPY32 cores C 3.7GHz 256 GB
177 Insanity
8x NVIDIA A10 24 GB
8 9216 CUDA cores
AMD Zen2 EPYC 3.3GHz
192 cores 768 GB
168 High-end studio 2
NVIDIA A10 24 GB
9216 CUDA cores
AMD Zen2 EPYC 3.3GHz
16 cores 64 GB
170 Top-tier 2 NVIDIA A9216 CUD10 24 GBA cores AMD Zen2 EPY32 cores C 3.3GHz 128 GB
179 Mid-tier 2 NVIDIA L4 27424 CUDA cores4 GB AMD Zen3 EPY8 cores C 3.7GHz 32 GB
178 Lite Insanity 8x NVIDIA L4 28 7 424 CUDA cores4 GB AMD Zen3 EPY192 cores C 3.7GHz 768 GB
181 Mid-tier 4 NVIDIA L40S 48 GB18176 CUDA cores AMD Zen3 EPY8 cores C 3.7GHz 64 GB
180 Ludicrous 8x NVIDIA L40S 48 GB8 18176 CUDA cores AMD Zen3 EPY192 cores C 3.7GHz 1536 GB
os_type (string, required): Operating system. Use ubuntu 24 for Ubuntu 2 4
app_type (string, optional): Application type. options can be: base, adobe_ccaa
template_version (null): Always set to null
Response
The response includes a temporary workstation id that you'll use in the deployment step.

{
"success": true,
"message": "Temp workstation created successfully",
"status" 200,
"id" 1689,
"workstation_name": "Spark 1st workstation"
}
Example (cURL)
curl  X POST \
'https://api.prod.aapse1.sparkcloud.studio/api/supabase/workstation/store-temp-workstation?context=a
pp' \
H 'accept: */*' \
H ' Authorization: Bearer YOUR_ACCESS_TOKEN' \
H 'Content-Type: application/json' \
-d '{
"region": "ap-southeast-1",
"workstation_static_id" 163,
"os_type": "ubuntu24",
"app_type": "base",
"template_version": null
}'
3. Deploy Your Ubuntu 24 Workstation
Deploy the temporary workstation created in the previous step using the id from the response.

Endpoint
POST https://api.prod.aapse1.sparkcloud.studio/api/workstations/v
Request Headers
{
"Content-Type": "application/json",
"Authorization": "Bearer YOUR_ACCESS_TOKEN"
}
Request Body
{
"temp_workstation_id": "1689"
}
Parameters
temp_workstation_id (string, required): The temporary workstation ID received from the creation response.
Response
The response includes a workstation_id that you'll use for starting, stopping, and deleting the workstation.

{
"status": "success",
"message": "Workstation deployed successfully",
"workstation_id" 10
}
Example (cURL)
curl  X POST https://api.prod.aapse1.sparkcloud.studio/api/workstations/v2 \
H " Authorization: Bearer YOUR_ACCESS_TOKEN" \
H "Content-Type: application/json" \
-d '{
"temp_workstation_id" 1689
}'
4. Listing Workstations
Retrieve information about existing workstations in your account.

4.1 Get a Single Workstation by ID
Use this endpoint to fetch the details of a specific workstation using its workstation_id received after a
successful deployment.

Endpoint
GET https://api.prod.aapse1.sparkcloud.studio/api/workstations/{id}
Path parameter:
id (required): The workstation_id of the workstation you want to retrieve.
Request Headers
{
"Content-Type": "application/json",
"Authorization": "Bearer YOUR_ACCESS_TOKEN"
}
Example (cURL)
curl  X 'GET' \
'https://api.prod.aapse1.sparkcloud.studio/api/workstations/{id}' \
H 'accept: */*' \
H ' Authorization: Bearer YOUR_ACCESS_TOKEN'
Example Response
{
"status": "success",
"data": {
// workstation object..
},
"error": null
}
4.2 List All Workstations
Use this endpoint to retrieve a list of all workstations associated with your account.

Endpoint
GET https://api.prod.aapse1.sparkcloud.studio/api/workstations'
Request Headers
{
"Content-Type": "application/json",
"Authorization": "Bearer YOUR_ACCESS_TOKEN"
}
Example (cURL)
curl  X 'GET' \
'https://api.prod.aapse1.sparkcloud.studio/api/workstations' \
H 'accept: */*' \
H ' Authorization: Bearer YOUR_ACCESS_TOKEN'
Example Response (truncated)
{
"status": "success",
"data": {
"status": "success",
"data": {
"data": [
{
// workstation object...
}
],
"error": null
}
}
}
5. Starting and Stopping a Workstation
Starting a Workstation
Start a stopped workstation to resume work. Use the workstation_id received during deployment.

Endpoint
POST https://api.prod.aapse1.sparkcloud.studio/api/workstations/{id}/start
Request Headers
{
"Authorization": "Bearer YOUR_ACCESS_TOKEN"
}
Response
{
"status": "success",
"message": "Workstation is starting"
}
Example (cURL)
curl  X POST https://api.prod.aapse1.sparkcloud.studio/api/workstations/10/start \
H " Authorization: Bearer YOUR_ACCESS_TOKEN"
Stopping a Workstation
Stop a running workstation to save resources.

Endpoint
POST https://api.prod.aapse1.sparkcloud.studio/api/workstations/{id}/stop
Request Headers
{
"Content-Type": "application/json",
"Authorization": "Bearer YOUR_ACCESS_TOKEN"
}
Response
{
"status": "success",
"message": "Workstation is stopping"
}
Example (cURL)
curl  X POST https://api.prod.aapse1.sparkcloud.studio/api/workstations/10/stop \
H " Authorization: Bearer YOUR_ACCESS_TOKEN"
6. Deleting a Workstation
Permanently delete a workstation. This action cannot be undone.

Endpoint
DELETE https://api.prod.aapse1.sparkcloud.studio/api/workstations/{id}
Request Headers
{
"Authorization": "Bearer YOUR_ACCESS_TOKEN"
}
Response
{
"message": "Workstation archived and deleted successfully",
"status": "success"
}
Example (cURL)
curl  X DELETE https://api.prod.aapse1.sparkcloud.studio/api/workstations/10 \
H " Authorization: Bearer YOUR_ACCESS_TOKEN"
Warning: Deleting a workstation will permanently remove all data stored on it.

Status Codes
The API uses standard HTTP status codes:

Status Code Description
200 OK Request successful
201 Created Resource created successfully
400 Bad Request Invalid request parameters
401 Unauthorized Invalid or missing access token
403 Forbidden Insufficient permissions
404 Not Found Resource not found
429 Too Many Requests Rate limit exceeded
500 Internal Server Error Server error
Support
For additional help or questions about the Spark Cloud Studio API, please contact:

Email: support@sparkcloud.studio