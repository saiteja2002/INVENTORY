import hashlib
import requests
import sys


def request_api(hash_first_5):
    #sends the request to api
    url = 'https://api.pwnedpasswords.com/range/' + hash_first_5
    response = requests.get(url)
    if response.status_code != 200:
        print("Error not working")
        raise RuntimeError(f'Error in API fetching, status {response.status_code}. Check the API and try again')
    return response


def database_password_count(response_data, reaming_5):
    api_response = (line.split(':') for line in response_data.splitlines())
    for h, count in api_response:
        if h == reaming_5:
            return int(count)
    return 0


def number_of_times_found_in_database(password):
    sha1_encoding = hashlib.sha1(password.encode('utf-8')).hexdigest().upper()#hahses the given password using sha1hashing
    first5chars, remainig_5 = sha1_encoding[:5], sha1_encoding[5:]#splits into first5 and last 5
    response = request_api(first5chars)#sends a request to api using first 5 characters of the hashed password
    count = database_password_count(response.text, remainig_5)#finds the number of times the password was in the reponse
    return count


def main(given_passwords):
    for password in given_passwords:
        count = number_of_times_found_in_database(password)
        if count:
            print(f'{password} was found {count} times in databse. You should probably change your password')
        else:
            print(f'{password} was not found in the dataBase. No need to change ')
    return 'Done'


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Use sys.exit() only to display return message from the main()
        # function at end of work
        sys.exit(main(sys.argv[1:]))
    else:
        pass
