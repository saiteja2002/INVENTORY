import requests
import hashlib
import sys

#requests to api with first 5 characters of password encoded in sha1 and recieves all the sha1 encoded passwords
#which have the same first 5 characters

def request_api_data(query_char):
  url = 'https://api.pwnedpasswords.com/range/' + query_char
  res = requests.get(url)
  if res.status_code != 200:
    raise RuntimeError(f'Error fetching: {res.status_code}, check the api and try again')
  return res


# returns number of times the given passwords have been been hacked
def get_password_leaks_count(hashes, hash_to_check):
  hashes = (line.split(':') for line in hashes.text.splitlines())
  for h, count in hashes:
    if h == hash_to_check:
      return count
  return 0
# encodes the given password with sha1 and calls the above functions
def check(password):
  sha1password = hashlib.sha1(password.encode('utf-8')).hexdigest().upper()
  first5_char, tail = sha1password[:5], sha1password[5:]
  response = request_api_data(first5_char)
  return get_password_leaks_count(response, tail)
#displays the count
def completed(args):
  for password in args:
    count = check(password)
    if count:
      print(f'{password} was found {count} times... change your password!')
    else:
      print(f'{password} was NOT found. You are safe!')
  return 'done!'


sys.exit(completed(sys.argv[1:]))

