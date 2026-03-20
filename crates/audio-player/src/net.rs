use crate::error::{Error, Result};

/// Rejects URLs that resolve to private, loopback, or link-local hosts.
///
/// This is a best-effort SSRF mitigation — it catches literal private IPs and
/// `localhost`, but cannot prevent DNS rebinding attacks. A full mitigation
/// would require resolving the hostname and checking the IP *after* DNS lookup
/// but *before* connecting, which ureq does not support.
pub(crate) fn reject_private_host(url: &str) -> Result<()> {
   let after_scheme = url.split("://").nth(1).unwrap_or("");
   let authority = after_scheme.split('/').next().unwrap_or("");
   // Strip userinfo (e.g. "user:pass@host").
   let host_and_port = authority.rsplit('@').next().unwrap_or(authority);

   // Parse the host, accounting for bracket-wrapped IPv6.
   let (host, bare) = if let Some(rest) = host_and_port.strip_prefix('[') {
      // IPv6: everything between '[' and ']' is the address.
      match rest.find(']') {
         Some(end) => (&host_and_port[..end + 2], &rest[..end]),
         None => (host_and_port, host_and_port),
      }
   } else {
      // IPv4 or hostname: strip port after the last colon.
      let h = host_and_port.split(':').next().unwrap_or("");
      (h, h)
   };

   if host.is_empty() {
      return Err(Error::Http("Invalid URL: empty host".into()));
   }

   if host.eq_ignore_ascii_case("localhost") {
      return Err(Error::Http("Requests to localhost are not allowed".into()));
   }

   if let Ok(ip) = bare.parse::<std::net::IpAddr>() {
      let blocked = match ip {
         std::net::IpAddr::V4(v4) => {
            v4.is_loopback()
               || v4.is_private()
               || v4.is_link_local()
               || v4.is_unspecified()
               || v4.is_broadcast()
         }
         std::net::IpAddr::V6(v6) => v6.is_loopback() || v6.is_unspecified(),
      };

      if blocked {
         return Err(Error::Http(format!(
            "Requests to private/reserved address {ip} are not allowed"
         )));
      }
   }

   Ok(())
}

#[cfg(test)]
mod tests {
   use super::*;

   fn expect_allowed(url: &str) {
      assert!(reject_private_host(url).is_ok(), "expected allowed: {url}");
   }

   fn expect_blocked(url: &str) {
      assert!(reject_private_host(url).is_err(), "expected blocked: {url}");
   }

   // -- Allowed URLs --
   #[test]
   fn allows_public_https() {
      expect_allowed("https://example.com/audio.mp3");
   }

   #[test]
   fn allows_public_http() {
      expect_allowed("http://cdn.example.com:8080/file");
   }

   #[test]
   fn allows_public_ip() {
      expect_allowed("https://93.184.216.34/audio.mp3");
   }

   #[test]
   fn allows_url_with_path_and_query() {
      expect_allowed("https://example.com/path?q=1");
   }

   #[test]
   fn allows_non_ip_hostname() {
      expect_allowed("http://internal.corp/file");
   }

   // -- Blocked: localhost --
   #[test]
   fn blocks_localhost() {
      expect_blocked("http://localhost/file");
   }

   #[test]
   fn blocks_localhost_case_insensitive() {
      expect_blocked("http://LOCALHOST/file");
   }

   #[test]
   fn blocks_localhost_with_port() {
      expect_blocked("http://localhost:3000/file");
   }

   // -- Blocked: loopback --
   #[test]
   fn blocks_ipv4_loopback() {
      expect_blocked("http://127.0.0.1/file");
   }

   #[test]
   fn blocks_ipv4_loopback_range() {
      expect_blocked("http://127.255.0.1/file");
   }

   #[test]
   fn blocks_ipv6_loopback() {
      expect_blocked("http://[::1]/file");
   }

   // -- Blocked: private RFC 1918 --
   #[test]
   fn blocks_10_x() {
      expect_blocked("http://10.0.0.1/file");
   }

   #[test]
   fn blocks_172_16_x() {
      expect_blocked("http://172.16.0.1/file");
   }

   #[test]
   fn blocks_192_168_x() {
      expect_blocked("http://192.168.1.1/file");
   }

   // -- Blocked: link-local --
   #[test]
   fn blocks_link_local() {
      expect_blocked("http://169.254.169.254/metadata");
   }

   // -- Blocked: unspecified / broadcast --
   #[test]
   fn blocks_unspecified() {
      expect_blocked("http://0.0.0.0/file");
   }

   #[test]
   fn blocks_broadcast() {
      expect_blocked("http://255.255.255.255/file");
   }

   #[test]
   fn blocks_ipv6_unspecified() {
      expect_blocked("http://[::]/file");
   }

   // -- Blocked: empty host --
   #[test]
   fn blocks_empty_host() {
      expect_blocked("http:///file");
   }

   // -- Userinfo stripping --
   #[test]
   fn strips_userinfo_and_still_blocks() {
      expect_blocked("http://user:pass@127.0.0.1/file");
   }

   #[test]
   fn strips_userinfo_and_allows_public() {
      expect_allowed("http://user@example.com/file");
   }
}
