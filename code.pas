function code(s:string;m:boolean):string;
var x:byte;mode:integer;st:string;
begin
  if m then st:='ccc' else st:='';
  if m then mode:=1 else mode:=-1;
  for x:=1 to length(s) do st:=st+chr(ord(s[x])+mode);
  code:=st;
end;
var st:string;
begin
  write('INPUT: ');readln(st);
  if copy(st,1,3)='ccc' then
  write('OUTPUT: ',code(copy(st,4,length(ST)-3),false))
  else write('OUTPUT: ',code(st,true));
  readln
end.
