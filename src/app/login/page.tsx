export default function Login() {
    return (
      <div>
        <h1 className="text-3xl font-bold mb-4">Login</h1>
        <form className="max-w-sm mx-auto">
          <div className="mb-4">
            <label className="block mb-2">Email</label>
            <input type="email" className="w-full p-2 border rounded" />
          </div>
          <div className="mb-4">
            <label className="block mb-2">Password</label>
            <input type="password" className="w-full p-2 border rounded" />
          </div>
          <button type="submit" className="bg-blue-500 text-white p-2 rounded">
            Login
          </button>
        </form>
      </div>
    );
  }