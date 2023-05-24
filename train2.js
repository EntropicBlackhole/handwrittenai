function fermatTest(n, k) {
	if (n <= 1) {
		return false;
	}

	// Perform Fermat's primality test 'k' times
	for (let i = 0; i < k; i++) {
		// Generate a random number between 2 and n-1
		const a = Math.floor(Math.random() * (n - 2)) + 2;

		// Calculate a^(n-1) mod n
		const result = modularExponentiation(a, n - 1, n);

		// If the result is not 1, then 'n' is composite
		if (result !== 1) {
			return false;
		}
	}

	// If 'n' passes the test 'k' times, it is likely prime
	return true;
}

// Function to perform modular exponentiation
function modularExponentiation(base, exponent, modulus) {
	let result = 1;
	base = base % modulus;

	while (exponent > 0) {
		if (exponent % 2 === 1) {
			result = (result * base) % modulus;
		}

		exponent = Math.floor(exponent / 2);
		base = (base * base) % modulus;
	}

	return result;
}
console.log(fermat(2, 3))